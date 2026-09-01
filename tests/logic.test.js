const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function loadApp() {
  const stub = {
    value: 'MyModel',
    classList: { add() {}, remove() {} },
    innerHTML: '',
    style: {},
    textContent: '',
    checked: false,
    disabled: false,
    children: [],
    appendChild() {},
    setAttribute() {},
    getContext() { return {}; },
    querySelector() { return null; },
  };

  const context = {
    window: {},
    document: {
      getElementById: () => stub,
      querySelectorAll: () => [],
      querySelector: () => null,
      createElement: () => ({
        click() {},
        setAttribute() {},
        style: {},
        href: '',
        download: '',
        type: '',
        className: '',
      }),
      addEventListener() {},
    },
    navigator: { clipboard: { writeText: async () => {} } },
    URL: { createObjectURL: () => 'blob://test', revokeObjectURL() {} },
    Blob: globalThis.Blob || function Blob() {},
    Image: function Image() {},
    XMLSerializer: function XMLSerializer() { return { serializeToString: () => '' }; },
    console,
    setTimeout,
    clearTimeout,
    performance: { now: () => 0 },
    requestAnimationFrame: (fn) => { fn(0); return 1; },
    cancelAnimationFrame: () => {},
    tf: null,
    activeCodeFramework: 'pytorch',
    layers: [],
    skips: [],
    saved: [],
    customLs: [],
    nextId: 1,
    selIdx: null,
  };

  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'script.js'), 'utf8'), context);
  return context;
}

function setModelState(app, layers, skips = []) {
  vm.runInContext(`layers = ${JSON.stringify(layers)}; skips = ${JSON.stringify(skips)};`, app);
}

test('validation rejects conv-to-dense without flatten', () => {
  const app = loadApp();
  setModelState(app, [
    { type: 'Conv2d', params: { in_channels: 3, out_channels: 8, kernel_size: 3, stride: 1, padding: 1 } },
    { type: 'Linear', params: { in_features: 8, out_features: 10, bias: true } },
  ]);

  const issues = app.validate();
  assert.ok(issues.some((issue) => issue.sev === 'error' && issue.msg.includes('Flatten')));
});

test('architecture export excludes training loop', () => {
  const app = loadApp();
  setModelState(app, [
    { type: 'Conv2d', params: { in_channels: 3, out_channels: 8, kernel_size: 3, stride: 1, padding: 1 } },
    { type: 'Flatten', params: {} },
    { type: 'Linear', params: { in_features: 8, out_features: 10, bias: true } },
  ]);
  app.document.getElementById = () => ({ ...stubDefaults(), value: 'MyModel' });

  const code = app.genCode('pytorch', false);
  assert.doesNotMatch(code, /for epoch in range\(/);
  assert.doesNotMatch(code, /TRAINING_CONFIG/);
});

test('training export includes training config and loop', () => {
  const app = loadApp();
  setModelState(app, [
    { type: 'Conv2d', params: { in_channels: 3, out_channels: 8, kernel_size: 3, stride: 1, padding: 1 } },
    { type: 'Flatten', params: {} },
    { type: 'Linear', params: { in_features: 8, out_features: 10, bias: true } },
  ]);
  app.document.getElementById = () => ({ ...stubDefaults(), value: 'MyModel' });

  const code = app.genCode('pytorch', true);
  assert.match(code, /TRAINING_CONFIG/);
  assert.match(code, /for epoch in range\(/);
  assert.match(code, /optimizer = torch\.optim\./);
});

test('tensorflow training export includes model and optimizer setup', () => {
  const app = loadApp();
  setModelState(app, [
    { type: 'Conv2d', params: { in_channels: 3, out_channels: 8, kernel_size: 3, stride: 1, padding: 1 } },
    { type: 'Flatten', params: {} },
    { type: 'Linear', params: { in_features: 8, out_features: 10, bias: true } },
  ]);
  app.document.getElementById = () => ({ ...stubDefaults(), value: 'MyModel' });

  const code = app.genCode('tensorflow', true);
  assert.match(code, /class MyModel\(tf\.keras\.Model\):/);
  assert.match(code, /optimizer = tf\.keras\.optimizers\./);
  assert.match(code, /for epoch in range\(/);
});

test('skip direction validation catches backward residuals', () => {
  const app = loadApp();
  setModelState(app, [
    { type: 'Linear', params: { in_features: 4, out_features: 6, bias: true } },
    { type: 'Linear', params: { in_features: 6, out_features: 8, bias: true } },
  ], [{ f: 1, t: 0 }]);

  const issues = app.validate();
  assert.ok(issues.some((issue) => issue.sev === 'error' && issue.msg.includes('backward')));
});

test('valid conv+flatten+linear passes validation', () => {
  const app = loadApp();
  setModelState(app, [
    { type: 'Conv2d', params: { in_channels: 3, out_channels: 8, kernel_size: 3, stride: 1, padding: 1 } },
    { type: 'Flatten', params: {} },
    { type: 'Linear', params: { in_features: 8, out_features: 10, bias: true } },
  ]);

  const issues = app.validate();
  assert.ok(!issues.some((issue) => issue.sev === 'error'));
});

test('parameter count matches a simple linear layer', () => {
  const app = loadApp();
  setModelState(app, [
    { type: 'Linear', params: { in_features: 3, out_features: 4, bias: true } },
  ]);

  assert.equal(app.countParams(), 16);
});

function stubDefaults() {
  return {
    value: 'MyModel',
    classList: { add() {}, remove() {} },
    innerHTML: '',
    style: {},
    textContent: '',
    checked: false,
    disabled: false,
    children: [],
    appendChild() {},
    setAttribute() {},
    getContext() { return {}; },
    querySelector() { return null; },
  };
}
