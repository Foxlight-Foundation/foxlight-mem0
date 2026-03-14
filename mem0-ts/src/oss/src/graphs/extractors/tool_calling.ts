import {
  GraphExtractor,
  GraphExtractorDeps,
  EntityTypeMap,
  Relationship,
} from "./types";
import {
  EXTRACT_ENTITIES_TOOL,
  RELATIONS_TOOL,
  DELETE_MEMORY_TOOL_GRAPH,
} from "../tools";
import { EXTRACT_RELATIONS_PROMPT, getDeleteMessages } from "../utils";
import { LLM } from "../../llms/base";
import { logger } from "../../utils/logger";

interface Tool {
  type: string;
  function: {
    name: string;
    description: string;
    parameters: Record<string, any>;
  };
}

const normalizeEntities = (
  entities: Array<{
    source: string;
    relationship: string;
    destination: string;
  }>,
) =>
  entities.map((item) => ({
    ...item,
    source: item.source.toLowerCase().replace(/ /g, "_"),
    relationship: item.relationship.toLowerCase().replace(/ /g, "_"),
    destination: item.destination.toLowerCase().replace(/ /g, "_"),
  }));

export class ToolCallingExtractor implements GraphExtractor {
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
    const tools = [EXTRACT_ENTITIES_TOOL] as Tool[];

    const defaultEntityPrompt = `You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use ${filters["userId"]} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.`;

    const entityPrompt = this.customEntityPrompt
      ? `${this.customEntityPrompt}\n\n${defaultEntityPrompt}`
      : defaultEntityPrompt;

    const searchResults = await this.llm.generateResponse(
      [
        { role: "system", content: entityPrompt },
        { role: "user", content: data },
      ],
      { type: "json_object" },
      tools,
    );

    let entityTypeMap: EntityTypeMap = {};
    try {
      if (typeof searchResults !== "string" && searchResults.toolCalls) {
        for (const call of searchResults.toolCalls) {
          if (call.name === "extract_entities") {
            const args = JSON.parse(call.arguments);
            for (const item of args.entities) {
              entityTypeMap[item.entity] = item.entity_type;
            }
          }
        }
      }
    } catch (e) {
      logger.error(`Error in search tool: ${e}`);
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
            "\nPlease provide your response in JSON format.",
        },
        { role: "user", content: data },
      ];
    } else {
      messages = [
        {
          role: "system",
          content:
            EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters["userId"]) +
            "\nPlease provide your response in JSON format.",
        },
        {
          role: "user",
          content: `List of entities: ${Object.keys(entityTypeMap)}. \n\nText: ${data}`,
        },
      ];
    }

    const tools = [RELATIONS_TOOL] as Tool[];
    const extractedEntities = await this.llm.generateResponse(
      messages,
      { type: "json_object" },
      tools,
    );

    let entities: Relationship[] = [];
    if (typeof extractedEntities !== "string" && extractedEntities.toolCalls) {
      const toolCall = extractedEntities.toolCalls[0];
      if (toolCall && toolCall.arguments) {
        const args = JSON.parse(toolCall.arguments);
        entities = args.entities || [];
      }
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

    const tools = [DELETE_MEMORY_TOOL_GRAPH] as Tool[];
    const memoryUpdates = await this.llm.generateResponse(
      [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      { type: "json_object" },
      tools,
    );

    const toBeDeleted: Relationship[] = [];
    if (typeof memoryUpdates !== "string" && memoryUpdates.toolCalls) {
      for (const item of memoryUpdates.toolCalls) {
        if (item.name === "delete_graph_memory") {
          toBeDeleted.push(JSON.parse(item.arguments));
        }
      }
    }

    const cleaned = normalizeEntities(toBeDeleted);
    logger.debug(`Deleted relationships: ${JSON.stringify(cleaned)}`);
    return cleaned;
  };
}
