export async function api<T>(path: string, data?: unknown): Promise<T> {
  const response = await fetch(path, {
    signal: AbortSignal.timeout(30_000),
    ...(data === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        }),
  });
  const value = await response.json();
  if (!response.ok)
    throw Object.assign(
      new Error(value.error || `Request failed: ${response.status}`),
      { status: response.status },
    );
  return value as T;
}
