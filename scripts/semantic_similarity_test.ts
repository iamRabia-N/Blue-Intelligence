import { embedTexts, cosineSimilarity } from "../lib/semanticDedup";

async function main() {
  const titleA = "Coral Reef Restoration Maldives";
  const titleB = "Maldives Coral Reef Recovery";
  const titleC = "Mangrove Conservation in Indonesia";

  const descriptionA = "A project focused on restoring coral reefs through community-driven reef planting.";
  const descriptionB = "Community-led efforts to recover coral reefs and marine biodiversity in the Maldives.";
  const descriptionC = "Protecting mangroves and coastal wetlands to support biodiversity and local communities.";

  const [embA, embB, embC] = await embedTexts([titleA, titleB, titleC]);
  const [descA, descB, descC] = await embedTexts([descriptionA, descriptionB, descriptionC]);

  console.log("Title similarity (A vs B):", cosineSimilarity(embA, embB).toFixed(4));
  console.log("Title similarity (A vs C):", cosineSimilarity(embA, embC).toFixed(4));
  console.log("Description similarity (A vs B):", cosineSimilarity(descA, descB).toFixed(4));
  console.log("Description similarity (A vs C):", cosineSimilarity(descA, descC).toFixed(4));

  console.log("\nSuggested thresholds: >= 0.85 for semantically matching projects.\n");
}

main().catch((e) => {
  console.error("Error running semantic similarity test:", e);
  process.exit(1);
});
