// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

import { RateLimiter, RateLimitedFunction, RateLimitOptions } from "../lancedb/util";

describe("Rate Limiting", () => {
  describe("RateLimiter", () => {
    it("should allow requests within rate limit", () => {
      const rateLimiter = new RateLimiter({
        maxRequests: 5,
        windowMs: 1000,
      });

      // Should allow 5 requests
      for (let i = 0; i < 5; i++) {
        expect(rateLimiter.tryConsume()).toBe(true);
      }

      // 6th request should be denied
      expect(rateLimiter.tryConsume()).toBe(false);
    });

    it("should refill tokens over time", async () => {
      const rateLimiter = new RateLimiter({
        maxRequests: 2,
        windowMs: 100, // Very short window for testing
      });

      // Consume all tokens
      expect(rateLimiter.tryConsume()).toBe(true);
      expect(rateLimiter.tryConsume()).toBe(true);
      expect(rateLimiter.tryConsume()).toBe(false);

      // Wait for refill
      await new Promise(resolve => setTimeout(resolve, 150));

      // Should be able to consume again
      expect(rateLimiter.tryConsume()).toBe(true);
    });

    it("should provide accurate token count", () => {
      const rateLimiter = new RateLimiter({
        maxRequests: 3,
        windowMs: 1000,
      });

      expect(rateLimiter.getAvailableTokens()).toBe(3);
      
      rateLimiter.tryConsume();
      expect(rateLimiter.getAvailableTokens()).toBe(2);
      
      rateLimiter.tryConsume(2);
      expect(rateLimiter.getAvailableTokens()).toBe(0);
    });

    it("should calculate time until next token", () => {
      const rateLimiter = new RateLimiter({
        maxRequests: 1,
        windowMs: 1000,
      });

      // Initially should have tokens available
      expect(rateLimiter.getTimeUntilNextToken()).toBe(0);

      // After consuming, should need to wait
      rateLimiter.tryConsume();
      expect(rateLimiter.getTimeUntilNextToken()).toBeGreaterThan(0);
    });

    it("should handle consume() with waiting", async () => {
      const rateLimiter = new RateLimiter({
        maxRequests: 1,
        windowMs: 100,
      });

      const start = Date.now();
      
      // First consume should be immediate
      await rateLimiter.consume();
      
      // Second consume should wait
      await rateLimiter.consume();
      
      const elapsed = Date.now() - start;
      expect(elapsed).toBeGreaterThan(50); // Should have waited some time
    });
  });

  describe("RateLimitedFunction", () => {
    it("should rate limit function calls", async () => {
      let callCount = 0;
      const mockFunction = async (value: string) => {
        callCount++;
        return `processed: ${value}`;
      };

      const rateLimitedFn = new RateLimitedFunction(mockFunction, {
        maxRequests: 2,
        windowMs: 100,
      });

      const start = Date.now();
      
      // First two calls should be fast
      await rateLimitedFn.execute("test1");
      await rateLimitedFn.execute("test2");
      
      // Third call should be rate limited
      const result = await rateLimitedFn.execute("test3");
      
      const elapsed = Date.now() - start;
      
      expect(callCount).toBe(3);
      expect(result).toBe("processed: test3");
      expect(elapsed).toBeGreaterThan(50); // Should have been delayed
    });

    it("should provide status information", async () => {
      const mockFunction = async () => "result";
      const rateLimitedFn = new RateLimitedFunction(mockFunction, {
        maxRequests: 5,
        windowMs: 1000,
      });

      // Initially should have full tokens
      let status = rateLimitedFn.getStatus();
      expect(status.availableTokens).toBe(5);
      expect(status.timeUntilNextToken).toBe(0);

      // After execution, should have fewer tokens
      await rateLimitedFn.execute();
      status = rateLimitedFn.getStatus();
      expect(status.availableTokens).toBe(4);
    });

    it("should handle multiple concurrent calls", async () => {
      let callCount = 0;
      const mockFunction = async () => {
        callCount++;
        return callCount;
      };

      const rateLimitedFn = new RateLimitedFunction(mockFunction, {
        maxRequests: 2,
        windowMs: 100,
      });

      // Start multiple calls concurrently
      const promises = [
        rateLimitedFn.execute(),
        rateLimitedFn.execute(),
        rateLimitedFn.execute(),
      ];

      const results = await Promise.all(promises);
      
      expect(results).toEqual([1, 2, 3]);
      expect(callCount).toBe(3);
    });

    it("should work with different function signatures", async () => {
      const addFunction = async (a: number, b: number) => a + b;
      const rateLimitedAdd = new RateLimitedFunction(addFunction, {
        maxRequests: 10,
        windowMs: 1000,
      });

      const result = await rateLimitedAdd.execute(5, 3);
      expect(result).toBe(8);
    });
  });

  describe("Integration with delay", () => {
    it("should respect delayMs option", async () => {
      const rateLimiter = new RateLimiter({
        maxRequests: 10,
        windowMs: 1000,
        delayMs: 50, // Minimum 50ms delay between requests
      });

      const start = Date.now();
      
      // Even though we have tokens, should still delay
      await rateLimiter.consume();
      await rateLimiter.consume();
      
      const elapsed = Date.now() - start;
      expect(elapsed).toBeGreaterThan(40); // Should have delayed
    });
  });

  describe("Edge cases", () => {
    it("should handle zero maxRequests", () => {
      const rateLimiter = new RateLimiter({
        maxRequests: 0,
        windowMs: 1000,
      });

      expect(rateLimiter.tryConsume()).toBe(false);
      expect(rateLimiter.getAvailableTokens()).toBe(0);
    });

    it("should handle very large windowMs", () => {
      const rateLimiter = new RateLimiter({
        maxRequests: 1,
        windowMs: 1000000, // Very large window
      });

      rateLimiter.tryConsume();
      expect(rateLimiter.getTimeUntilNextToken()).toBeGreaterThan(500000);
    });

    it("should handle fractional tokens correctly", () => {
      const rateLimiter = new RateLimiter({
        maxRequests: 3,
        windowMs: 1000,
      });

      // Consume partial tokens
      expect(rateLimiter.tryConsume(0.5)).toBe(true);
      expect(rateLimiter.getAvailableTokens()).toBe(2.5);
      
      expect(rateLimiter.tryConsume(2.5)).toBe(true);
      expect(rateLimiter.getAvailableTokens()).toBe(0);
    });
  });
});
