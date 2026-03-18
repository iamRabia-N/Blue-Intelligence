// Type declarations for TensorFlow.js packages to suppress TypeScript warnings
declare module '@tensorflow-models/universal-sentence-encoder' {
  export function load(): Promise<{
    embed(texts: string[]): Promise<number[][]>;
  }>;
}

declare module '@tensorflow/tfjs-node' {
  // Empty declaration to satisfy import
}