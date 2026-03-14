import {
  GraphExtractor,
  GraphExtractorDeps,
  EntityTypeMap,
  Relationship,
} from "./types";
import {
  GraphExtractEntitiesArgsSchema,
  GraphRelationsArgsSchema,
  GraphSimpleRelationshipArgsSchema,
  ENTITY_TYPES,
} from "../tools";
import { EXTRACT_RELATIONS_PROMPT, getDeleteMessages } from "../utils";
import { LLM } from "../../llms/base";
import { logger } from "../../utils/logger";
import { z } from "zod";

const stripMarkdownFences = (text: string): string =>
  text.replace(/^```(?:json)?\s*\n?/i, "").replace(/\n?```\s*$/i, "");

const parseJsonResponse = (response: any): string => {
  const raw = typeof response === "string" ? response : (response?.content ?? "");
  return stripMarkdownFences(raw.trim());
};

const normalizeEntities = (
  entities: Array<{ source: string; relationship: string; destination: string }>,
) =>
  entities.map((item) => ({
    ...item,
    source: item.source.toLowerCase().replace(/ /g, "_"),
    relationship: item.relationship.toLowerCase().replace(/ /g, "_"),
    destination: item.destination.toLowerCase().replace(/ /g, "_"),
  }));

const ENTITY_SCHEMA_PROMPT = `
Respond with ONLY valid JSON matching this exact schema — no markdown, no explanation:
{
  "entities": [
    {
      "entity": "<name of the entity>",
      "entity_type": "<one of: ${ENTITY_TYPES.join(", ")}>"
    }
  ]
}`;

const RELATIONS_SCHEMA_PROMPT = `
Respond with ONLY valid JSON matching this exact schema — no markdown, no explanation:
{
  "entities": [
    {
      "source": "<source entity>",
      "relationship": "<relationship type>",
      "destination": "<destination entity>"
    }
  ]
}`;

const DELETIONS_SCHEMA_PROMPT = `
Respond with ONLY valid JSON matching this exact schema — no markdown, no explanation.
If no relationships should be deleted, respond with { "deletions": [] }.
{
  "deletions": [
    {
      "source": "<source entity>",
      "relationship": "<relationship type>",
      "destination": "<destination entity>"
    }
  ]
}`;

export class JsonPromptExtractor implements GraphExtractor {
  private llm: LLM;
  private customPrompt?: string;
  private customEntityPrompt?: string;

  constructor(deps: GraphExtractorDeps) {
    this.llm = deps.llm;
    this.customPrompt = deps.customPrompt;
    this.customEntityPrompt = deps.customEntityPrompt;
  }

  extractEntities = async (
    data: string,
    filters: Record<string, any>,
  ): Promise<EntityTypeMap> => {
    const defaultEntityPrompt = `You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use ${filters["userId"]} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.`;

    const entityPrompt = this.customEntityPrompt
      ? `${this.customEntityPrompt}\n\n${defaultEntityPrompt}`
      : defaultEntityPrompt;

    const response = await this.llm.generateResponse(
      [
        {
          role: "system",
          content: `${entityPrompt}\n${ENTITY_SCHEMA_PROMPT}`,
        },
        { role: "user", content: data },
      ],
      { type: "json_object" },
    );

    let entityTypeMap: EntityTypeMap = {};
    try {
      const raw = parseJsonResponse(response);
      const parsed = JSON.parse(raw);
      const result = GraphExtractEntitiesArgsSchema.safeParse(parsed);
      if (result.success) {
        for (const item of result.data.entities) {
          entityTypeMap[item.entity] = item.entity_type;
        }
      } else {
        logger.warn(
          `JSON entity extraction failed validation: ${result.error.message}`,
        );
      }
    } catch (e) {
      logger.error(`Error parsing JSON entity response: ${e}`);
    }

    entityTypeMap = Object.fromEntries(
      Object.entries(entityTypeMap).map(([k, v]) => [
        k.toLowerCase().replace(/ /g, "_"),
        v.toLowerCase().replace(/ /g, "_"),
      ]),
    );

    logger.debug(`Entity type map: ${JSON.stringify(entityTypeMap)}`);
    return entityTypeMap;
  };

  extractRelationships = async (
    data: string,
    filters: Record<string, any>,
    entityTypeMap: EntityTypeMap,
  ): Promise<Relationship[]> => {
    let messages;
    if (this.customPrompt) {
      messages = [
        {
          role: "system",
          content:
            EXTRACT_RELATIONS_PROMPT.replace(
              "USER_ID",
              filters["userId"],
            ).replace("CUSTOM_PROMPT", `4. ${this.customPrompt}`) +
            `\n${RELATIONS_SCHEMA_PROMPT}`,
        },
        { role: "user", content: data },
      ];
    } else {
      messages = [
        {
          role: "system",
          content:
            EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters["userId"]) +
            `\n${RELATIONS_SCHEMA_PROMPT}`,
        },
        {
          role: "user",
          content: `List of entities: ${Object.keys(entityTypeMap)}. \n\nText: ${data}`,
        },
      ];
    }

    const response = await this.llm.generateResponse(messages, {
      type: "json_object",
    });

    let entities: Relationship[] = [];
    try {
      const raw = parseJsonResponse(response);
      const parsed = JSON.parse(raw);
      const result = GraphRelationsArgsSchema.safeParse(parsed);
      if (result.success) {
        entities = result.data.entities;
      } else {
        logger.warn(
          `JSON relationship extraction failed validation: ${result.error.message}`,
        );
      }
    } catch (e) {
      logger.error(`Error parsing JSON relationship response: ${e}`);
    }

    entities = normalizeEntities(entities);
    logger.debug(`Extracted entities: ${JSON.stringify(entities)}`);
    return entities;
  };

  extractDeletions = async (
    existingTriples: Relationship[],
    data: string,
    filters: Record<string, any>,
  ): Promise<Relationship[]> => {
    const searchOutputString = existingTriples
      .map(
        (item) =>
          `${item.source} -- ${item.relationship} -- ${item.destination}`,
      )
      .join("\n");

    const [systemPrompt, userPrompt] = getDeleteMessages(
      searchOutputString,
      data,
      filters["userId"],
    );

    const response = await this.llm.generateResponse(
      [
        {
          role: "system",
          content: `${systemPrompt}\n${DELETIONS_SCHEMA_PROMPT}`,
        },
        { role: "user", content: userPrompt },
      ],
      { type: "json_object" },
    );

    const toBeDeleted: Relationship[] = [];
    try {
      const raw = parseJsonResponse(response);
      const parsed = JSON.parse(raw);
      const result = z
        .object({
          deletions: z.array(GraphSimpleRelationshipArgsSchema),
        })
        .safeParse(parsed);
      if (result.success) {
        toBeDeleted.push(...result.data.deletions);
      } else {
        logger.warn(
          `JSON deletion extraction failed validation: ${result.error.message}`,
        );
      }
    } catch (e) {
      logger.error(`Error parsing JSON deletion response: ${e}`);
    }

    const cleaned = normalizeEntities(toBeDeleted);
    logger.debug(`Deleted relationships: ${JSON.stringify(cleaned)}`);
    return cleaned;
  };
}
