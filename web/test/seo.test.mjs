// The page's search-engine plumbing stays intact: canonical, social cards,
// structured data that parses, crawl files that agree with each other.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { chartEntries } from "../js/reference.js";

const WEB = join(dirname(fileURLToPath(import.meta.url)), "..");
const SITE = "https://jetzhu.github.io/MorseCodeAudioCoder/";
const html = readFileSync(join(WEB, "index.html"), "utf8");

const attr = (re) => {
  const m = re.exec(html);
  return m ? m[1] : null;
};

test("head carries a canonical URL, a description and social cards", () => {
  assert.equal(attr(/<link rel="canonical" href="([^"]+)"/), SITE);
  const description = attr(/<meta name="description" content="([^"]+)"/);
  assert.ok(description && description.length >= 100 && description.length <= 220, description);
  assert.equal(attr(/<meta property="og:url" content="([^"]+)"/), SITE);
  assert.equal(attr(/<meta property="og:image" content="([^"]+)"/), `${SITE}og.png`);
  assert.equal(attr(/<meta name="twitter:card" content="([^"]+)"/), "summary_large_image");
  const title = attr(/<title>([^<]+)<\/title>/);
  for (const term of ["Morse Code Simulator", "Decoder", "Trainer"]) assert.ok(title.includes(term), `${term} in the title`);
  assert.ok(title.length <= 75, `title length ${title.length}`);
  for (const term of ["simulator", "translate", "practise", "chart"]) assert.ok(description.includes(term), `${term} in the description`);
  assert.equal((html.match(/<h1[\s>]/g) || []).length, 1, "exactly one h1");
  assert.ok(existsSync(join(WEB, "og.png")));
});

test("structured data parses and describes the app and the FAQ", () => {
  const blocks = [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/g)].map((m) => JSON.parse(m[1]));
  assert.equal(blocks.length, 2);
  const graph = blocks[0]["@graph"];
  assert.equal(graph[0]["@type"], "WebApplication");
  assert.equal(graph[0].url, SITE);
  assert.equal(graph[1]["@type"], "SoftwareApplication");
  const faq = blocks[1];
  assert.equal(faq["@type"], "FAQPage");
  assert.ok(faq.mainEntity.length >= 4);
  for (const q of faq.mainEntity) {
    assert.ok(html.includes(`<h3>${q.name}</h3>`), `FAQ question on the page: ${q.name}`);
  }
});

test("robots.txt, sitemap.xml and the mock agree", () => {
  const robots = readFileSync(join(WEB, "robots.txt"), "utf8");
  assert.ok(robots.includes(`Sitemap: ${SITE}sitemap.xml`));
  assert.ok(robots.includes("Disallow: /mock.html"));
  const sitemap = readFileSync(join(WEB, "sitemap.xml"), "utf8");
  assert.ok(sitemap.includes(`<loc>${SITE}</loc>`));
  assert.ok(/<lastmod>\d{4}-\d{2}-\d{2}<\/lastmod>/.test(sitemap));
  const mock = readFileSync(join(WEB, "mock.html"), "utf8");
  assert.ok(mock.includes('<meta name="robots" content="noindex,nofollow">'));
});

test("the chart is in the HTML and matches the shared table", () => {
  const cells = [...html.matchAll(/<button type="button" class="refcell" data-code="([^"]+)" title="Play [^"]*"><span class="ch">([^<]+)<\/span>/g)];
  const entries = chartEntries();
  assert.equal(cells.length, entries.length);
  cells.forEach((m, i) => {
    const ch = m[2].replace(/&quot;/g, '"').replace(/&amp;/g, "&");
    assert.equal(ch, entries[i][0]);
    assert.equal(m[1], entries[i][1]);
  });
  assert.ok(html.includes("Morse code simulator"));
  assert.ok(/<h3>Is this a Morse code simulator\?<\/h3>/.test(html));
});
