// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

import { expect, test } from "@jest/globals";
import * as lancedb from "@lancedb/lancedb";
import "@lancedb/lancedb/embedding/openai";
import { LanceSchema, getRegistry, EmbeddingFunction } from "@lancedb/lancedb/embedding";
import { Float32, Utf8 } from "apache-arrow";
import { withTempDirectory } from "./util.ts";

const openAiTest = process.env.OPENAI_API_KEY == null ? test.skip : test;

openAiTest("openai embeddings with rate limiting", async () => {
  await withTempDirectory(async (databaseDir) => {
    // --8<-- [start:openai_rate_limiting]
    const db = await lancedb.connect(databaseDir);
    
    // Create OpenAI embedding function with rate limiting
    const func = getRegistry()
      .get("openai")
      ?.create({ 
        model: "text-embedding-ada-002",
        rateLimit: {
          maxRequests: 3000, // OpenAI tier 1 limit
          windowMs: 60000,   // 1 minute window
          delayMs: 100,      // Optional: minimum delay between requests
        }
      }) as EmbeddingFunction;

    const wordsSchema = LanceSchema({
      text: func.sourceField(new Utf8()),
      vector: func.vectorField(),
    });
    
    const tbl = await db.createEmptyTable("words", wordsSchema, {
      mode: "overwrite",
    });
    
    // Add data - rate limiting will be applied automatically
    await tbl.add([
      { text: "hello world" }, 
      { text: "goodbye world" },
      { text: "rate limited request" },
    ]);

    // Check rate limiter status
    const status = func.getRateLimitStatus();
    console.log("Rate limit status:", status);

    const query = "greetings";
    const actual = (await tbl.search(query).limit(1).toArray())[0];
    // --8<-- [end:openai_rate_limiting]
    expect(actual).toHaveProperty("text");
  });
});

test("custom embedding function with rate limiting", async () => {
  await withTempDirectory(async (databaseDir) => {
    // --8<-- [start:custom_rate_limiting]
    const db = await lancedb.connect(databaseDir);

    class CustomAPIEmbeddingFunction extends EmbeddingFunction<string> {
      private apiKey: string;
      private callCount = 0;

      constructor(options: { 
        apiKey: string; 
        rateLimit?: { maxRequests: number; windowMs: number; delayMs?: number } 
      }) {
        super();
        const resolvedOptions = this.resolveVariables(options);
        this.apiKey = resolvedOptions.apiKey;
      }

      ndims() {
        return 384; // Example dimension
      }

      embeddingDataType() {
        return new Float32();
      }

      async computeSourceEmbeddings(data: string[]): Promise<number[][]> {
        // Simulate API call with rate limiting
        const apiCall = async () => {
          this.callCount++;
          console.log(`API call #${this.callCount} for ${data.length} texts`);
          
          // Simulate API response delay
          await new Promise(resolve => setTimeout(resolve, 50));
          
          // Return mock embeddings
          return data.map(() => Array.from({ length: 384 }, () => Math.random()));
        };

        // This will apply rate limiting if configured
        return this.executeWithRateLimit(apiCall);
      }

      async computeQueryEmbeddings(data: string): Promise<number[]> {
        const apiCall = async () => {
          this.callCount++;
          console.log(`Query API call #${this.callCount} for: "${data}"`);
          
          await new Promise(resolve => setTimeout(resolve, 50));
          return Array.from({ length: 384 }, () => Math.random());
        };

        return this.executeWithRateLimit(apiCall);
      }

      getCallCount() {
        return this.callCount;
      }
    }

    // Create function with rate limiting
    const func = new CustomAPIEmbeddingFunction({
      apiKey: "test-key",
      rateLimit: {
        maxRequests: 2,    // Only 2 requests
        windowMs: 1000,    // Per second
        delayMs: 200,      // With 200ms minimum delay
      },
    });

    const data = [
      { text: "first text" },
      { text: "second text" },
      { text: "third text" },  // This will be rate limited
    ];

    const start = Date.now();
    
    const table = await db.createTable("custom_rate_limited", data, {
      embeddingFunction: {
        function: func,
        sourceColumn: "text",
        vectorColumn: "vector",
      },
      mode: "overwrite",
    });

    const elapsed = Date.now() - start;
    
    // Should have taken time due to rate limiting
    console.log(`Operation took ${elapsed}ms`);
    console.log(`Total API calls: ${func.getCallCount()}`);
    
    // Check rate limiter status
    const status = func.getRateLimitStatus();
    console.log("Final rate limit status:", status);
    
    // Verify data was inserted
    const count = await table.countRows();
    expect(count).toBe(3);
    
    // The operation should have been delayed due to rate limiting
    expect(elapsed).toBeGreaterThan(400); // At least some delay
    // --8<-- [end:custom_rate_limiting]
  });
});

test("rate limiting with different configurations", async () => {
  await withTempDirectory(async (databaseDir) => {
    // --8<-- [start:rate_limit_configs]
    const db = await lancedb.connect(databaseDir);

    class ConfigurableEmbeddingFunction extends EmbeddingFunction<string> {
      constructor(options: any = {}) {
        super();
        this.resolveVariables(options);
      }

      ndims() { return 3; }
      embeddingDataType() { return new Float32(); }

      async computeSourceEmbeddings(data: string[]): Promise<number[][]> {
        const apiCall = async () => {
          await new Promise(resolve => setTimeout(resolve, 10));
          return data.map(() => [1, 2, 3]);
        };
        return this.executeWithRateLimit(apiCall);
      }

      async computeQueryEmbeddings(data: string): Promise<number[]> {
        const apiCall = async () => {
          await new Promise(resolve => setTimeout(resolve, 10));
          return [1, 2, 3];
        };
        return this.executeWithRateLimit(apiCall);
      }
    }

    // Test different rate limiting configurations
    const configs = [
      {
        name: "Conservative",
        rateLimit: { maxRequests: 1, windowMs: 500 }
      },
      {
        name: "Moderate", 
        rateLimit: { maxRequests: 5, windowMs: 1000 }
      },
      {
        name: "Aggressive",
        rateLimit: { maxRequests: 10, windowMs: 1000, delayMs: 50 }
      }
    ];

    for (const config of configs) {
      const func = new ConfigurableEmbeddingFunction(config);
      
      const start = Date.now();
      
      // Make multiple calls
      await func.computeQueryEmbeddings("test1");
      await func.computeQueryEmbeddings("test2");
      await func.computeQueryEmbeddings("test3");
      
      const elapsed = Date.now() - start;
      console.log(`${config.name} config took ${elapsed}ms`);
      
      const status = func.getRateLimitStatus();
      console.log(`${config.name} final status:`, status);
    }
    // --8<-- [end:rate_limit_configs]
  });
});

test("rate limiting error handling", async () => {
  await withTempDirectory(async (databaseDir) => {
    // --8<-- [start:rate_limit_errors]
    class ErrorHandlingEmbeddingFunction extends EmbeddingFunction<string> {
      constructor(options: any = {}) {
        super();
        this.resolveVariables(options);
      }

      ndims() { return 3; }
      embeddingDataType() { return new Float32(); }

      async computeSourceEmbeddings(data: string[]): Promise<number[][]> {
        const apiCall = async () => {
          // Simulate API errors
          if (Math.random() < 0.3) {
            throw new Error("Simulated API error");
          }
          return data.map(() => [1, 2, 3]);
        };

        try {
          return await this.executeWithRateLimit(apiCall);
        } catch (error) {
          console.log("API call failed:", error.message);
          // Return fallback embeddings
          return data.map(() => [0, 0, 0]);
        }
      }

      async computeQueryEmbeddings(data: string): Promise<number[]> {
        const apiCall = async () => {
          if (Math.random() < 0.3) {
            throw new Error("Simulated API error");
          }
          return [1, 2, 3];
        };

        try {
          return await this.executeWithRateLimit(apiCall);
        } catch (error) {
          console.log("Query API call failed:", error.message);
          return [0, 0, 0];
        }
      }
    }

    const func = new ErrorHandlingEmbeddingFunction({
      rateLimit: {
        maxRequests: 3,
        windowMs: 1000,
      },
    });

    const db = await lancedb.connect(databaseDir);
    
    // This should handle errors gracefully
    const table = await db.createTable("error_handling", [
      { text: "test1" },
      { text: "test2" },
      { text: "test3" },
    ], {
      embeddingFunction: {
        function: func,
        sourceColumn: "text",
      },
      mode: "overwrite",
    });

    const count = await table.countRows();
    expect(count).toBe(3);
    // --8<-- [end:rate_limit_errors]
  });
});
