import neo4j, { Driver } from "neo4j-driver";
import { BM25 } from "../utils/bm25";
import { MemoryConfig } from "../types";
import { EmbedderFactory, LLMFactory } from "../utils/factory";
import { Embedder } from "../embeddings/base";
import {
  createGraphExtractor,
  GraphExtractor,
} from "../graphs/extractors";
import { logger } from "../utils/logger";

interface SearchOutput {
  source: string;
  source_id: string;
  relationship: string;
  relation_id: string;
  destination: string;
  destination_id: string;
  similarity: number;
}

interface GraphMemoryResult {
  deleted_entities: any[];
  added_entities: any[];
  relations?: any[];
  added_node_ids: string[];
  added_edge_ids: string[];
  deleted_edge_ids: string[];
}

export class MemoryGraph {
  private config: MemoryConfig;
  private graph: Driver;
  private embeddingModel: Embedder;
  private llmProvider: string;
  private extractor: GraphExtractor;
  private database?: string;
  private threshold: number;
  private nodeDeduplicationThreshold: number;
  private bm25TopK: number;

  constructor(config: MemoryConfig) {
    this.config = config;
    if (
      !config.graphStore?.config?.url ||
      !config.graphStore?.config?.username ||
      !config.graphStore?.config?.password
    ) {
      throw new Error("Neo4j configuration is incomplete");
    }

    this.graph = neo4j.driver(
      config.graphStore.config.url,
      neo4j.auth.basic(
        config.graphStore.config.username,
        config.graphStore.config.password,
      ),
    );

    this.embeddingModel = EmbedderFactory.create(
      this.config.embedder.provider,
      this.config.embedder.config,
    );

    this.llmProvider = "openai";
    if (this.config.llm?.provider) {
      this.llmProvider = this.config.llm.provider;
    }
    if (this.config.graphStore?.llm?.provider) {
      this.llmProvider = this.config.graphStore.llm.provider;
    }

    // Use graphStore-specific LLM config when provided, fall back to main LLM config.
    // This is what makes MEM0_GRAPH_LLM_MODEL take effect.
    const graphLlmConfig =
      this.config.graphStore?.llm?.config ?? this.config.llm.config;
    const graphLlm = LLMFactory.create(this.llmProvider, graphLlmConfig);

    this.extractor = createGraphExtractor(
      config.graphStore?.extractionStrategy ?? "tool_calling",
      {
        llm: graphLlm,
        customPrompt: config.graphStore?.customPrompt,
        customEntityPrompt: config.graphStore?.customEntityPrompt,
      },
    );

    this.database = config.graphStore?.config?.database;
    this.threshold = config.graphStore?.searchThreshold ?? 0.7;
    this.nodeDeduplicationThreshold =
      config.graphStore?.nodeDeduplicationThreshold ?? 0.9;
    this.bm25TopK = config.graphStore?.bm25TopK ?? 5;
  }

  async add(
    data: string,
    filters: Record<string, any>,
  ): Promise<GraphMemoryResult> {
    const entityTypeMap = await this._retrieveNodesFromData(data, filters);

    const toBeAdded = await this._establishNodesRelationsFromData(
      data,
      filters,
      entityTypeMap,
    );

    const searchOutput = await this._searchGraphDb(
      Object.keys(entityTypeMap),
      filters,
    );

    const toBeDeleted = await this._getDeleteEntitiesFromSearchOutput(
      searchOutput,
      data,
      filters,
    );

    const { records: deletedEntities, edgeIds: deletedEdgeIds } =
      await this._deleteEntities(toBeDeleted, filters["userId"]);

    const {
      records: addedEntities,
      nodeIds: addedNodeIds,
      edgeIds: addedEdgeIds,
    } = await this._addEntities(toBeAdded, filters["userId"], entityTypeMap);

    return {
      deleted_entities: deletedEntities,
      added_entities: addedEntities,
      relations: toBeAdded,
      added_node_ids: addedNodeIds,
      added_edge_ids: addedEdgeIds,
      deleted_edge_ids: deletedEdgeIds,
    };
  }

  async search(query: string, filters: Record<string, any>, limit = 100) {
    const entityTypeMap = await this._retrieveNodesFromData(query, filters);
    const searchOutput = await this._searchGraphDb(
      Object.keys(entityTypeMap),
      filters,
    );

    if (!searchOutput.length) {
      return [];
    }

    const searchOutputsSequence = searchOutput.map((item) => [
      item.source,
      item.relationship,
      item.destination,
    ]);

    const bm25 = new BM25(searchOutputsSequence);
    const tokenizedQuery = query.split(" ");
    const rerankedResults = bm25.search(tokenizedQuery).slice(0, this.bm25TopK);

    const searchResults = rerankedResults.map((item) => ({
      source: item[0],
      relationship: item[1],
      destination: item[2],
    }));

    logger.info(`Returned ${searchResults.length} search results`);
    return searchResults;
  }

  async deleteAll(filters: Record<string, any>) {
    const session = this.graph.session({ database: this.database });
    try {
      await session.run("MATCH (n {user_id: $user_id}) DETACH DELETE n", {
        user_id: filters["userId"],
      });
    } finally {
      await session.close();
    }
  }

  async getAll(filters: Record<string, any>, limit = 100) {
    const session = this.graph.session({ database: this.database });
    try {
      const result = await session.run(
        `
        MATCH (n {user_id: $user_id})-[r]->(m {user_id: $user_id})
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT toInteger($limit)
        `,
        { user_id: filters["userId"], limit: Math.floor(Number(limit)) },
      );

      const finalResults = result.records.map((record) => ({
        source: record.get("source"),
        relationship: record.get("relationship"),
        target: record.get("target"),
      }));

      logger.info(`Retrieved ${finalResults.length} relationships`);
      return finalResults;
    } finally {
      await session.close();
    }
  }

  private async _retrieveNodesFromData(
    data: string,
    filters: Record<string, any>,
  ) {
    return this.extractor.extractEntities(data, filters);
  }

  private async _establishNodesRelationsFromData(
    data: string,
    filters: Record<string, any>,
    entityTypeMap: Record<string, string>,
  ) {
    return this.extractor.extractRelationships(data, filters, entityTypeMap);
  }

  private async _searchGraphDb(
    nodeList: string[],
    filters: Record<string, any>,
    limit = 100,
  ): Promise<SearchOutput[]> {
    const resultRelations: SearchOutput[] = [];
    const session = this.graph.session({ database: this.database });

    try {
      for (const node of nodeList) {
        const nEmbedding = await this.embeddingModel.embed(node);

        const cypher = `
          MATCH (n)
          WHERE n.embedding IS NOT NULL AND n.user_id = $user_id
          WITH n,
              round(reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | dot + n.embedding[i] * $n_embedding[i]) /
              (sqrt(reduce(l2 = 0.0, i IN range(0, size(n.embedding)-1) | l2 + n.embedding[i] * n.embedding[i])) *
              sqrt(reduce(l2 = 0.0, i IN range(0, size($n_embedding)-1) | l2 + $n_embedding[i] * $n_embedding[i]))), 4) AS similarity
          WHERE similarity >= $threshold
          MATCH (n)-[r]->(m)
          RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id, similarity
          UNION
          MATCH (n)
          WHERE n.embedding IS NOT NULL AND n.user_id = $user_id
          WITH n,
              round(reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | dot + n.embedding[i] * $n_embedding[i]) /
              (sqrt(reduce(l2 = 0.0, i IN range(0, size(n.embedding)-1) | l2 + n.embedding[i] * n.embedding[i])) *
              sqrt(reduce(l2 = 0.0, i IN range(0, size($n_embedding)-1) | l2 + $n_embedding[i] * $n_embedding[i]))), 4) AS similarity
          WHERE similarity >= $threshold
          MATCH (m)-[r]->(n)
          RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id, similarity
          ORDER BY similarity DESC
          LIMIT toInteger($limit)
        `;

        const result = await session.run(cypher, {
          n_embedding: nEmbedding,
          threshold: this.threshold,
          user_id: filters["userId"],
          limit: Math.floor(Number(limit)),
        });

        resultRelations.push(
          ...result.records.map((record) => ({
            source: record.get("source"),
            source_id: record.get("source_id").toString(),
            relationship: record.get("relationship"),
            relation_id: record.get("relation_id").toString(),
            destination: record.get("destination"),
            destination_id: record.get("destination_id").toString(),
            similarity: record.get("similarity"),
          })),
        );
      }
    } finally {
      await session.close();
    }

    return resultRelations;
  }

  private async _getDeleteEntitiesFromSearchOutput(
    searchOutput: SearchOutput[],
    data: string,
    filters: Record<string, any>,
  ) {
    return this.extractor.extractDeletions(
      searchOutput.map((item) => ({
        source: item.source,
        relationship: item.relationship,
        destination: item.destination,
      })),
      data,
      filters,
    );
  }

  private async _deleteEntities(toBeDeleted: any[], userId: string) {
    const results: any[] = [];
    const edgeIds: string[] = [];
    const session = this.graph.session({ database: this.database });

    try {
      for (const item of toBeDeleted) {
        const { source, destination, relationship } = item;

        const cypher = `
          MATCH (n {name: $source_name, user_id: $user_id})
          -[r:${relationship}]->
          (m {name: $dest_name, user_id: $user_id})
          WITH n, r, m, elementId(r) AS rel_id
          DELETE r
          RETURN
              n.name AS source,
              m.name AS target,
              type(r) AS relationship,
              rel_id
        `;

        const result = await session.run(cypher, {
          source_name: source,
          dest_name: destination,
          user_id: userId,
        });

        for (const record of result.records) {
          const relId = record.get("rel_id");
          if (relId) edgeIds.push(String(relId));
        }
        results.push(result.records);
      }
    } finally {
      await session.close();
    }

    return { records: results, edgeIds };
  }

  private async _addEntities(
    toBeAdded: any[],
    userId: string,
    entityTypeMap: Record<string, string>,
  ): Promise<{ records: any[]; nodeIds: string[]; edgeIds: string[] }> {
    const results: any[] = [];
    const nodeIds: string[] = [];
    const edgeIds: string[] = [];
    const session = this.graph.session({ database: this.database });

    try {
      for (const item of toBeAdded) {
        const { source, destination, relationship } = item;
        const sourceType = entityTypeMap[source] || "concept";
        const destinationType = entityTypeMap[destination] || "concept";
        if (!entityTypeMap[source]) {
          logger.warn(
            `Entity "${source}" not in entity map, defaulting to "concept"`,
          );
        }
        if (!entityTypeMap[destination]) {
          logger.warn(
            `Entity "${destination}" not in entity map, defaulting to "concept"`,
          );
        }

        const sourceEmbedding = await this.embeddingModel.embed(source);
        const destEmbedding = await this.embeddingModel.embed(destination);

        const sourceNodeSearchResult = await this._searchSourceNode(
          sourceEmbedding,
          userId,
          this.nodeDeduplicationThreshold,
        );
        const destinationNodeSearchResult = await this._searchDestinationNode(
          destEmbedding,
          userId,
          this.nodeDeduplicationThreshold,
        );

        let cypher: string;
        let params: Record<string, any>;

        if (
          destinationNodeSearchResult.length === 0 &&
          sourceNodeSearchResult.length > 0
        ) {
          cypher = `
            MATCH (source)
            WHERE elementId(source) = $source_id
            MERGE (destination:${destinationType} {name: $destination_name, user_id: $user_id})
            ON CREATE SET
                destination.created = timestamp(),
                destination.embedding = $destination_embedding
            MERGE (source)-[r:${relationship}]->(destination)
            ON CREATE SET
                r.created = timestamp()
            RETURN source.name AS source, type(r) AS relationship, destination.name AS target,
                   elementId(source) AS source_id, elementId(destination) AS dest_id, elementId(r) AS rel_id
          `;

          params = {
            source_id: sourceNodeSearchResult[0].elementId,
            destination_name: destination,
            destination_embedding: destEmbedding,
            user_id: userId,
          };
        } else if (
          destinationNodeSearchResult.length > 0 &&
          sourceNodeSearchResult.length === 0
        ) {
          cypher = `
            MATCH (destination)
            WHERE elementId(destination) = $destination_id
            MERGE (source:${sourceType} {name: $source_name, user_id: $user_id})
            ON CREATE SET
                source.created = timestamp(),
                source.embedding = $source_embedding
            MERGE (source)-[r:${relationship}]->(destination)
            ON CREATE SET
                r.created = timestamp()
            RETURN source.name AS source, type(r) AS relationship, destination.name AS target,
                   elementId(source) AS source_id, elementId(destination) AS dest_id, elementId(r) AS rel_id
          `;

          params = {
            destination_id: destinationNodeSearchResult[0].elementId,
            source_name: source,
            source_embedding: sourceEmbedding,
            user_id: userId,
          };
        } else if (
          sourceNodeSearchResult.length > 0 &&
          destinationNodeSearchResult.length > 0
        ) {
          cypher = `
            MATCH (source)
            WHERE elementId(source) = $source_id
            MATCH (destination)
            WHERE elementId(destination) = $destination_id
            MERGE (source)-[r:${relationship}]->(destination)
            ON CREATE SET
                r.created_at = timestamp(),
                r.updated_at = timestamp()
            RETURN source.name AS source, type(r) AS relationship, destination.name AS target,
                   elementId(source) AS source_id, elementId(destination) AS dest_id, elementId(r) AS rel_id
          `;

          params = {
            source_id: sourceNodeSearchResult[0]?.elementId,
            destination_id: destinationNodeSearchResult[0]?.elementId,
            user_id: userId,
          };
        } else {
          cypher = `
            MERGE (n:${sourceType} {name: $source_name, user_id: $user_id})
            ON CREATE SET n.created = timestamp(), n.embedding = $source_embedding
            ON MATCH SET n.embedding = $source_embedding
            MERGE (m:${destinationType} {name: $dest_name, user_id: $user_id})
            ON CREATE SET m.created = timestamp(), m.embedding = $dest_embedding
            ON MATCH SET m.embedding = $dest_embedding
            MERGE (n)-[rel:${relationship}]->(m)
            ON CREATE SET rel.created = timestamp()
            RETURN n.name AS source, type(rel) AS relationship, m.name AS target,
                   elementId(n) AS source_id, elementId(m) AS dest_id, elementId(rel) AS rel_id
          `;

          params = {
            source_name: source,
            dest_name: destination,
            source_embedding: sourceEmbedding,
            dest_embedding: destEmbedding,
            user_id: userId,
          };
        }

        const result = await session.run(cypher, params);
        for (const record of result.records) {
          const srcId = record.get("source_id");
          const dstId = record.get("dest_id");
          const relId = record.get("rel_id");
          if (srcId && !nodeIds.includes(String(srcId)))
            nodeIds.push(String(srcId));
          if (dstId && !nodeIds.includes(String(dstId)))
            nodeIds.push(String(dstId));
          if (relId && !edgeIds.includes(String(relId)))
            edgeIds.push(String(relId));
        }
        results.push(result.records);
      }
    } finally {
      await session.close();
    }

    return { records: results, nodeIds, edgeIds };
  }

  private async _searchSourceNode(
    sourceEmbedding: number[],
    userId: string,
    threshold = 0.9,
  ) {
    const session = this.graph.session({ database: this.database });
    try {
      const cypher = `
        MATCH (source_candidate)
        WHERE source_candidate.embedding IS NOT NULL 
        AND source_candidate.user_id = $user_id

        WITH source_candidate,
            round(
                reduce(dot = 0.0, i IN range(0, size(source_candidate.embedding)-1) |
                    dot + source_candidate.embedding[i] * $source_embedding[i]) /
                (sqrt(reduce(l2 = 0.0, i IN range(0, size(source_candidate.embedding)-1) |
                    l2 + source_candidate.embedding[i] * source_candidate.embedding[i])) *
                sqrt(reduce(l2 = 0.0, i IN range(0, size($source_embedding)-1) |
                    l2 + $source_embedding[i] * $source_embedding[i])))
                , 4) AS source_similarity
        WHERE source_similarity >= $threshold

        WITH source_candidate, source_similarity
        ORDER BY source_similarity DESC
        LIMIT 1

        RETURN elementId(source_candidate) as element_id
        `;

      const params = {
        source_embedding: sourceEmbedding,
        user_id: userId,
        threshold,
      };

      const result = await session.run(cypher, params);

      return result.records.map((record) => ({
        elementId: record.get("element_id").toString(),
      }));
    } finally {
      await session.close();
    }
  }

  private async _searchDestinationNode(
    destinationEmbedding: number[],
    userId: string,
    threshold = 0.9,
  ) {
    const session = this.graph.session({ database: this.database });
    try {
      const cypher = `
        MATCH (destination_candidate)
        WHERE destination_candidate.embedding IS NOT NULL 
        AND destination_candidate.user_id = $user_id

        WITH destination_candidate,
            round(
                reduce(dot = 0.0, i IN range(0, size(destination_candidate.embedding)-1) |
                    dot + destination_candidate.embedding[i] * $destination_embedding[i]) /
                (sqrt(reduce(l2 = 0.0, i IN range(0, size(destination_candidate.embedding)-1) |
                    l2 + destination_candidate.embedding[i] * destination_candidate.embedding[i])) *
                sqrt(reduce(l2 = 0.0, i IN range(0, size($destination_embedding)-1) |
                    l2 + $destination_embedding[i] * $destination_embedding[i])))
            , 4) AS destination_similarity
        WHERE destination_similarity >= $threshold

        WITH destination_candidate, destination_similarity
        ORDER BY destination_similarity DESC
        LIMIT 1

        RETURN elementId(destination_candidate) as element_id
        `;

      const params = {
        destination_embedding: destinationEmbedding,
        user_id: userId,
        threshold,
      };

      const result = await session.run(cypher, params);

      return result.records.map((record) => ({
        elementId: record.get("element_id").toString(),
      }));
    } finally {
      await session.close();
    }
  }
}
