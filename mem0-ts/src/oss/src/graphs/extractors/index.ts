export type {
  GraphExtractor,
  GraphExtractorDeps,
  EntityTypeMap,
  Relationship,
} from "./types";
export { ToolCallingExtractor } from "./tool_calling";
export { JsonPromptExtractor } from "./json_prompting";

import { GraphExtractor, GraphExtractorDeps } from "./types";
import { ToolCallingExtractor } from "./tool_calling";
import { JsonPromptExtractor } from "./json_prompting";

export type ExtractionStrategy = "tool_calling" | "json_prompting";

export const createGraphExtractor = (
  strategy: ExtractionStrategy,
  deps: GraphExtractorDeps,
): GraphExtractor => {
  switch (strategy) {
    case "json_prompting":
      return new JsonPromptExtractor(deps);
    case "tool_calling":
    default:
      return new ToolCallingExtractor(deps);
  }
};
