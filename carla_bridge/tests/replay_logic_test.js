/*
 * Unit-test review.html's real application script in Node's VM, using only
 * owned, minimal DOM/media/canvas doubles. No browser, CDP, HTTP, or network.
 *
 * Manual use: node carla_bridge/tests/replay_logic_test.js
 */
'use strict';
if (Number(process.versions.node.split('.')[0]) < 16) {
  console.log('SKIP: optional replay logic tests require Node.js 16 or later.');
  process.exit(77);
}
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const template = fs.readFileSync(path.join(__dirname, '..', 'replay', 'review.html'), 'utf8');
const scripts = [...template.matchAll(/<script>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length, 1, 'test must execute the production application script');
const source = scripts[0][1];

class FakeNode {
  constructor(tagName, id = '') {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.children = [];
    this.ownText = '';
    this.className = '';
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this.listeners = new Map();
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.value = '';
    this.paused = true;
    this.currentTime = 0;
    this.width = 1000;
    this.height = 500;
    this.pauseCount = 0;
    this.loadCount = 0;
    this.canvasOperations = [];
    const self = this;
    this.classList = {
      add(...names) {
        const current = new Set(self.className.split(/\s+/).filter(Boolean));
        names.forEach(name => current.add(name));
        self.className = [...current].join(' ');
      },
      contains(name) { return self.className.split(/\s+/).includes(name); },
      toggle(name, force) {
        const enabled = force === undefined ? !this.contains(name) : Boolean(force);
        const current = new Set(self.className.split(/\s+/).filter(Boolean));
        if (enabled) current.add(name); else current.delete(name);
        self.className = [...current].join(' ');
        return enabled;
      },
    };
  }
  set textContent(value) { this.children = []; this.ownText = String(value); }
  get textContent() { return this.ownText + this.children.map(child => child.textContent).join(''); }
  // The production renderer must not convert model strings into HTML.
  set innerHTML(_) { throw new Error('Production code attempted to set innerHTML'); }
  append(...nodes) { this.children.push(...nodes); }
  prepend(...nodes) { this.children.unshift(...nodes); }
  replaceChildren(...nodes) { this.ownText = ''; this.children = [...nodes]; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  addEventListener(name, listener) {
    if (!this.listeners.has(name)) this.listeners.set(name, []);
    this.listeners.get(name).push(listener);
  }
  emit(name, args = {}) {
    for (const listener of this.listeners.get(name) || []) listener({target: this, ...args});
  }
  pause() { this.paused = true; this.pauseCount++; }
  load() { this.loadCount++; }
  getBoundingClientRect() { return {left: 0, top: 0, width: this.width, height: this.height}; }
  getContext(type) {
    assert.equal(type, '2d');
    const operations = this.canvasOperations;
    return Object.fromEntries(['clearRect', 'fillText', 'beginPath', 'moveTo', 'lineTo', 'stroke', 'arc', 'fill']
      .map(name => [name, (...args) => operations.push([name, ...args])]));
  }
}

function dataFixture() {
  const calls = [
    {frame: 500, time: 50, dispatch: true},
    {frame: 502, time: 50.2, dispatch: true}, // Intentionally not encoded.
    {frame: 504, time: 50.4, dispatch: false}, // Consultation before SDK dispatch.
    {frame: 505, time: 50.5, dispatch: true},
  ].map((item, i) => ({
    call_id: `call_${i}`, sample_token: `sample_${i}`,
    carla_frame_id: item.frame, sim_time_s: item.time,
    ego: {x: i * 10, y: i * 2, speed_mps: 5},
    completed: true, status: 'success', duration_wall_s: i,
    has_actual_sdk_dispatch: item.dispatch,
    actual_sdk_attempt_count: item.dispatch ? 2 : 0,
    trigger_text: `trigger_${i}`, trigger: `trigger_${i}`,
    images: [{path: 'model_call_assets/input.jpg', url: '../model_call_assets/input.jpg'}],
    record: {
      current_scene: {scene_description: `场景 ${i}`},
      raw_response: '</script><script>throw new Error("untrusted response")</script>',
      model_decision_output: {lateral_decision: 'FOLLOW_ROUTE'},
    },
    planner: {related_plan_count: 3, semantic_effect_plan_count: i, samples: []},
    requests: item.dispatch ? ['scene', 'decision'].map(stage => ({
      request_id: `request_${i}_${stage}`, attempt: 1, stage, status: 'success',
      prompt: `exact prompt ${i} ${stage}`, system_prompt: 'exact system prompt',
      response: `raw response ${i} ${stage}`, images: [], wall_duration_s: 0,
    })) : [],
  }));
  return {
    summary: {
      run_directory: 'VM_UNIT_FIXTURE', warnings: [], trigger_counts: {blocked_lane: 3},
      model_call_frame_ratio: .5, actual_model_call_frame_count: 3, total_tick_frame_count: 6,
      consultation_count: calls.length, sdk_attempt_count: 6, logical_model_request_count: 6,
      sdk_failed_attempt_count: 0, semantic_effect_plan_ratio: .2,
      semantic_effect_plan_count: 2, planner_plan_count: 10,
      semantic_effect_tick_frame_ratio: .2,
      semantic_effect_tick_frame_count: 1, semantic_attributable_tick_frame_count: 5,
    },
    run: {scenario_path: 'owned unit fixture', status: 'fixture'}, calls,
    track: Array.from({length: 6}, (_, i) => ({frame: 500 + i, sim_time_s: 50 + i / 10, x: i * 5, y: i})),
    videos: [{
      name: 'owned_fixture.mp4', url: '../carla_videos/owned_fixture.mp4',
      mapping_source: 'owned_fixture.frames.jsonl',
      // 20 fps encoding against 10 Hz simulation, a 50 s origin, and one gap.
      frames: [[0, 500, 50], [.05, 501, 50.1], [.10, 503, 50.3], [.15, 504, 50.4], [.20, 505, 50.5]],
    }],
  };
}

function boot(data = dataFixture(), useVideoFrameCallback = false) {
  const nodes = new Map();
  for (const match of template.matchAll(/<([a-zA-Z][\w-]*)\b[^>]*\bid="([^"]+)"/g)) {
    nodes.set(match[2], new FakeNode(match[1], match[2]));
  }
  const document = {
    getElementById(id) {
      assert(nodes.has(id), `unexpected DOM lookup: ${id}`);
      return nodes.get(id);
    },
    createElement(tagName) { return new FakeNode(tagName); },
  };
  nodes.get('review-data').textContent = JSON.stringify(data);
  nodes.get('follow').checked = true;
  const callbacks = [];
  if (useVideoFrameCallback) {
    nodes.get('video').requestVideoFrameCallback = callback => {
      callbacks.push(callback);
      return callbacks.length;
    };
  }
  // No fetch, XMLHttpRequest, browser globals, location, socket, or timers are
  // provided. Reading media sources cannot perform any IO through these doubles.
  const context = vm.createContext({document});
  vm.runInContext(source, context, {filename: 'review.html application script', timeout: 5000});
  return {
    node: id => nodes.get(id), callbacks,
    run: expression => vm.runInContext(expression, context, {timeout: 5000}),
  };
}

let passed = 0;
function test(name, callback) {
  callback();
  passed++;
  console.log(`PASS ${name}`);
}

test('initializes the production script with exact request text and zero durations', () => {
  const app = boot();
  assert.equal(app.run('selected'), 0);
  assert.equal(app.node('metrics').children.length, 5);
  assert.match(app.node('requests').textContent, /exact prompt 0 scene/);
  assert.match(app.node('requests').textContent, /raw response 0 decision/);
  assert.match(app.node('requests').textContent, /0\.00 墙钟秒/);
  assert.match(app.node('detailSummary').textContent, /耗时 0\.00 墙钟秒/);
  assert.match(app.node('callRecord').textContent, /<script>throw new Error/);
  assert.equal(app.node('video').loadCount, 1);
});

test('manual exact-frame seek uses encoded time, never simulation time', () => {
  const app = boot();
  app.run('selectCall(3, true)');
  assert.equal(app.run('selected'), 3);
  assert.equal(app.run('currentFrame'), 505);
  assert(Math.abs(app.node('video').currentTime - .20001) < 1e-10);
  assert.notEqual(app.node('video').currentTime, 50.5);
  assert.match(app.node('seekMessage').textContent, /精确定位 CARLA 帧 #505/);
  assert.match(app.node('instant').textContent, /触发瞬间.*#4/);
});

test('missing encoded frame stays selected while paused, including timeupdate', () => {
  const app = boot();
  app.run('selectCall(1, true)');
  assert.equal(app.run('currentFrame'), 501);
  assert.equal(app.run('selected'), 1);
  assert.equal(app.node('video').paused, true);
  assert(Math.abs(app.node('video').currentTime - .05001) < 1e-10);
  assert.match(app.node('seekMessage').textContent, /目标帧未录入视频.*相差 1 帧/);
  app.node('video').emit('timeupdate');
  assert.equal(app.run('selected'), 1, 'paused seek must not switch back to the preceding consultation');
  assert.match(app.node('detailTitle').textContent, /调用 2/);
  assert.doesNotMatch(app.node('instant').textContent, /触发瞬间/);
});

test('playback follows nearest preceding consultation only when enabled', () => {
  const app = boot();
  app.run('selectCall(1, true)');
  app.node('video').paused = false;
  app.node('video').currentTime = .15;
  app.node('video').emit('timeupdate');
  assert.equal(app.run('selected'), 2);
  assert.equal(app.run('currentFrame'), 504);
  assert.doesNotMatch(app.node('instant').textContent, /触发瞬间/);
  app.node('follow').checked = false;
  app.node('video').currentTime = .20;
  app.node('video').emit('timeupdate');
  assert.equal(app.run('selected'), 2, 'unchecked follow preserves user selection');
  assert.match(app.node('instant').textContent, /触发瞬间.*#4/);
  app.node('follow').checked = true;
  app.node('video').emit('timeupdate');
  assert.equal(app.run('selected'), 3);
});

test('exact invocation badge and row highlight exist only on actual SDK frames', () => {
  const app = boot();
  app.run('updateClock([0, 500, 50])');
  assert.match(app.node('instant').textContent, /触发瞬间.*#1/);
  assert(app.node('eventRows').children[0].classList.contains('triggerNow'));
  app.run('updateClock([.05, 501, 50.1])');
  assert.doesNotMatch(app.node('instant').textContent, /触发瞬间/);
  assert(!app.node('eventRows').children[0].classList.contains('triggerNow'));
  assert.match(app.node('instant').textContent, /仅作上下文/);
  app.run('updateClock([.15, 504, 50.4])');
  assert.doesNotMatch(app.node('instant').textContent, /触发瞬间/);
  assert(!app.node('eventRows').children[2].classList.contains('triggerNow'));
  assert(app.node('ruler').children[2].classList.contains('noDispatch'));
});

test('binary video-time lookup honors encoded gaps and differing fps', () => {
  const app = boot();
  assert.equal(app.run('frameBeforeVideoTime(.099)[1]'), 501);
  assert.equal(app.run('frameBeforeVideoTime(.10)[1]'), 503);
  assert.equal(app.run('frameBeforeVideoTime(.199)[1]'), 504);
  assert.equal(app.run('frameBeforeVideoTime(10)[1]'), 505);
});

test('unknown CARLA frame uses explicit nearest mapped simulation time with warning', () => {
  const app = boot();
  app.run('seekFrame(null, 50.31)');
  assert.equal(app.run('currentFrame'), 503);
  assert(Math.abs(app.node('video').currentTime - .10001) < 1e-10);
  assert.match(app.node('seekMessage').textContent, /缺少 CARLA 帧号.*近似对齐/);
});

test('absent mapping does not invent video seek or trigger-frame alignment', () => {
  const data = dataFixture();
  data.videos[0].frames = [];
  data.videos[0].mapping_source = null;
  const app = boot(data);
  app.node('video').currentTime = .123;
  app.run('selectCall(3, true)');
  assert.equal(app.node('video').currentTime, .123);
  assert.equal(app.run('currentFrame'), null);
  assert.equal(app.run('selected'), 3);
  assert.match(app.node('seekMessage').textContent, /没有逐帧时间映射.*未对视频做推测跳转/);
  assert.equal(app.node('noVideo').hidden, false);
});

test('no video retains call details and input-output review', () => {
  const data = dataFixture();
  data.videos = [];
  const app = boot(data);
  app.run('selectCall(3, true)');
  assert.equal(app.node('video').hidden, true);
  assert.equal(app.run('selected'), 3);
  assert.match(app.node('noVideo').textContent, /本轮没有驾驶视频/);
  assert.match(app.node('requests').textContent, /exact prompt 3 decision/);
});

test('empty model-free run initializes without calls, video, or coordinates', () => {
  const data = dataFixture();
  data.calls = []; data.track = []; data.videos = [];
  data.summary.consultation_count = 0; data.summary.trigger_counts = {};
  const app = boot(data);
  assert.equal(app.run('selected'), -1);
  assert.equal(app.node('previous').disabled, true);
  assert.equal(app.node('next').disabled, true);
  assert.equal(app.node('seekSelected').disabled, true);
  assert.match(app.node('detailSummary').textContent, /纯 PDM/);
  assert(app.node('map').canvasOperations.some(args => args[0] === 'fillText'));
  app.run('selectCall(0, true)');
  assert.equal(app.run('selected'), -1);
});

test('video playback without consultations remains reviewable', () => {
  const data = dataFixture();
  data.calls = [];
  data.summary.consultation_count = 0; data.summary.trigger_counts = {};
  const app = boot(data);
  app.node('video').paused = false;
  app.node('video').currentTime = .1;
  app.node('video').emit('timeupdate');
  assert.equal(app.run('currentFrame'), 503);
  assert.equal(app.run('selected'), -1);
  assert.match(app.node('instant').textContent, /当前帧没有新的模型调用/);
});

test('decoded-frame callback uses mediaTime, not wall clock or stale currentTime', () => {
  const app = boot(dataFixture(), true);
  assert.equal(app.callbacks.length, 1);
  app.node('video').paused = false;
  app.node('video').currentTime = 0;
  app.callbacks.shift()(999999, {mediaTime: .1});
  assert.equal(app.run('currentFrame'), 503);
  assert.equal(app.callbacks.length, 1, 'callback must schedule the next decoded-frame observation');
  assert.match(app.node('videoStatus').textContent, /编码视频 0\.10 s.*仿真 50\.30 s/);
});

test('next/previous UI handlers preserve missing-frame selection while paused', () => {
  const app = boot();
  app.node('next').emit('click');
  assert.equal(app.run('selected'), 1);
  assert.equal(app.run('currentFrame'), 501);
  app.node('next').emit('click');
  assert.equal(app.run('selected'), 2);
  app.node('previous').emit('click');
  assert.equal(app.run('selected'), 1);
  app.run('selectCall(-1, true); selectCall(500, true)');
  assert.equal(app.run('selected'), 1);
});

test('missing image handling reports the missing archive without network or markup', () => {
  const app = boot();
  const gallery = app.node('callImages').children[1];
  const figure = gallery.children[0];
  const image = figure.children[0].children[0];
  image.emit('error');
  assert.equal(image.hidden, true);
  assert.match(figure.textContent, /图片不可读取.*下载完整运行目录/);
});

console.log(`${passed} replay logic unit tests passed (Node VM + owned doubles; no browser or network).`);
