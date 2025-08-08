// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

export type IntoSql =
  | string
  | number
  | boolean
  | null
  | Date
  | ArrayBufferLike
  | Buffer
  | IntoSql[];

export function toSQL(value: IntoSql): string {
  if (typeof value === "string") {
    return `'${value.replace(/'/g, "''")}'`;
  } else if (typeof value === "number") {
    return value.toString();
  } else if (typeof value === "boolean") {
    return value ? "TRUE" : "FALSE";
  } else if (value === null) {
    return "NULL";
  } else if (value instanceof Date) {
    return `'${value.toISOString()}'`;
  } else if (Array.isArray(value)) {
    return `[${value.map(toSQL).join(", ")}]`;
  } else if (Buffer.isBuffer(value)) {
    return `X'${value.toString("hex")}'`;
  } else if (value instanceof ArrayBuffer) {
    return `X'${Buffer.from(value).toString("hex")}'`;
  } else {
    throw new Error(
      `Unsupported value type: ${typeof value} value: (${value})`,
    );
  }
}

export function packBits(data: Array<number>): Array<number> {
  const packed = Array(data.length >> 3).fill(0);
  for (let i = 0; i < data.length; i++) {
    const byte = i >> 3;
    const bit = i & 7;
    packed[byte] |= data[i] << bit;
  }
  return packed;
}

export class TTLCache {
  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  private readonly cache: Map<string, { value: any; expires: number }>;

  /**
   * @param ttl Time to live in milliseconds
   */
  constructor(private readonly ttl: number) {
    this.cache = new Map();
  }

  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  get(key: string): any | undefined {
    const entry = this.cache.get(key);
    if (entry === undefined) {
      return undefined;
    }

    if (entry.expires < Date.now()) {
      this.cache.delete(key);
      return undefined;
    }

    return entry.value;
  }

  // biome-ignore lint/suspicious/noExplicitAny: <explanation>
  set(key: string, value: any): void {
    this.cache.set(key, { value, expires: Date.now() + this.ttl });
  }

  delete(key: string): void {
    this.cache.delete(key);
  }
}

/**
 * Rate limiting configuration options
 */
export interface RateLimitOptions {
  /** Maximum number of requests per time window */
  maxRequests: number;
  /** Time window in milliseconds */
  windowMs: number;
  /** Optional delay between requests in milliseconds */
  delayMs?: number;
}

/**
 * Token bucket rate limiter implementation
 */
export class RateLimiter {
  private tokens: number;
  private lastRefill: number;
  private readonly maxTokens: number;
  private readonly refillRate: number; // tokens per millisecond
  private readonly delayMs: number;

  constructor(options: RateLimitOptions) {
    this.maxTokens = options.maxRequests;
    this.tokens = this.maxTokens;
    this.lastRefill = Date.now();
    this.refillRate = options.maxRequests / options.windowMs;
    this.delayMs = options.delayMs || 0;
  }

  /**
   * Attempt to consume a token. Returns true if successful, false if rate limited.
   */
  tryConsume(tokensRequested = 1): boolean {
    this.refillTokens();
    
    if (this.tokens >= tokensRequested) {
      this.tokens -= tokensRequested;
      return true;
    }
    
    return false;
  }

  /**
   * Wait until tokens are available, then consume them
   */
  async consume(tokensRequested = 1): Promise<void> {
    while (!this.tryConsume(tokensRequested)) {
      // Calculate wait time until next token is available
      const tokensNeeded = tokensRequested - this.tokens;
      const waitTime = Math.max(tokensNeeded / this.refillRate, this.delayMs);
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }
  }

  /**
   * Get the current number of available tokens
   */
  getAvailableTokens(): number {
    this.refillTokens();
    return this.tokens;
  }

  /**
   * Get time until next token is available (in milliseconds)
   */
  getTimeUntilNextToken(): number {
    this.refillTokens();
    if (this.tokens >= 1) {
      return 0;
    }
    return (1 - this.tokens) / this.refillRate;
  }

  private refillTokens(): void {
    const now = Date.now();
    const timePassed = now - this.lastRefill;
    const tokensToAdd = timePassed * this.refillRate;
    
    this.tokens = Math.min(this.maxTokens, this.tokens + tokensToAdd);
    this.lastRefill = now;
  }
}

/**
 * Rate-limited function wrapper that applies rate limiting to async functions
 */
export class RateLimitedFunction<T extends (...args: any[]) => Promise<any>> {
  private rateLimiter: RateLimiter;
  private originalFunction: T;

  constructor(fn: T, options: RateLimitOptions) {
    this.originalFunction = fn;
    this.rateLimiter = new RateLimiter(options);
  }

  /**
   * Execute the function with rate limiting applied
   */
  async execute(...args: Parameters<T>): Promise<Awaited<ReturnType<T>>> {
    await this.rateLimiter.consume();
    return this.originalFunction(...args);
  }

  /**
   * Get rate limiter status
   */
  getStatus() {
    return {
      availableTokens: this.rateLimiter.getAvailableTokens(),
      timeUntilNextToken: this.rateLimiter.getTimeUntilNextToken(),
    };
  }
}
