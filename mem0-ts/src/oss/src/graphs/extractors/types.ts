import { LLM } from "../../llms/base";

export interface EntityTypeMap {
  [entity: string]: string;
}

export interface Relationship {
  source: string;
  relationship: string;
  destination: string;
}

export interface GraphExtractorDeps {
  llm: LLM;
  customPrompt?: string;
  customEntityPrompt?: string;
}

export interface GraphExtractor {
  extractEntities: (
    data: string,
    filters: Record<string, any>,
  ) => Promise<EntityTypeMap>;

  extractRelationships: (
    data: string,
    filters: Record<string, any>,
    entityTypeMap: EntityTypeMap,
  ) => Promise<Relationship[]>;

  extractDeletions: (
    existingTriples: Relationship[],
    data: string,
    filters: Record<string, any>,
  ) => Promise<Relationship[]>;
}
