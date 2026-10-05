// WASM ORT can't handle concurrent session.run() calls: chain them all here.
let tail: Promise<unknown> = Promise.resolve();

export function serialRun<T>(fn: () => Promise<T>): Promise<T> {
  const p = tail.then(fn, fn);
  tail = p.catch(() => {});
  return p;
}
