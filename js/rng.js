export class RNG {
  constructor(seed = 123456789) {
    this.s = seed >>> 0 || 1;
  }

  next() {
    this.s += 0x6D2B79F5;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  normal() {
    const u = Math.max(this.next(), 1e-9);
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  int(n) {
    if (n <= 0) return 0;
    return Math.floor(this.next() * n);
  }

  pick(arr) {
    return arr[this.int(arr.length)];
  }
}

export function fillGauss(arr, std, rng) {
  for (let i = 0; i < arr.length; i++) arr[i] = rng.normal() * std;
}
