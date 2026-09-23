# Getting the page found

What is already done on the page and the repository is listed at the end of
`docs/PLAN.md` ("Search engines"). The steps below need the owner's accounts.
Each takes a few minutes; do them in this order.

Page: <https://jetzhu.github.io/MorseCodeAudioCoder/>
Sitemap: <https://jetzhu.github.io/MorseCodeAudioCoder/sitemap.xml>

## 1. Google Search Console (the one that matters most)

1. Open <https://search.google.com/search-console> and sign in with a Google
   account.
2. Add property. Choose **URL prefix** (not Domain, since the domain is
   github.io) and enter `https://jetzhu.github.io/MorseCodeAudioCoder/`.
3. Under verification methods pick **HTML tag**. Copy the line it shows,
   which looks like
   `<meta name="google-site-verification" content="XXXX">`.
4. In `web/index.html`, find the comment
   `Search Console and Bing Webmaster verification tags go here` in the head
   and paste the tag on the line after it. Commit and push; wait for the
   Pages deploy (about a minute; the Actions tab shows it).
5. Back in Search Console press **Verify**.
6. Left menu, **Sitemaps**: enter `sitemap.xml` and submit.
7. Top search bar (URL inspection): paste the page URL, then **Request
   indexing**. This is the fastest way to get the first crawl.
8. Come back after two weeks. **Performance** shows the queries that brought
   impressions and clicks; if one wording wins, move it forward in the title.

## 2. Bing Webmaster Tools

Bing, DuckDuckGo (which uses Bing's index), Yandex and others have already
received the page through IndexNow, which the repository submits with the key
file `web/<key>.txt` (`python tools/indexnow.py` resubmits after a change;
the first submission on 2026-09-23 was accepted with HTTP 202). Verification still helps, and it gives you Bing's own
reports:

1. Open <https://www.bing.com/webmasters> and sign in (a Microsoft account).
2. Choose **Import from Google Search Console** if step 1 is done; it copies
   the property and verification in one click. Otherwise **Add a site
   manually**, enter the page URL, pick the **HTML meta tag** method and paste
   the `<meta name="msvalidate.01" ...>` tag next to the Google one in
   `web/index.html`; push and verify.
3. **Sitemaps**: submit `https://jetzhu.github.io/MorseCodeAudioCoder/sitemap.xml`.

## 3. GitHub profile and repository

1. Pin the repository: on <https://github.com/jetzhu>, **Customize your pins**,
   tick MorseCodeAudioCoder. Pinned repositories are crawled from the profile
   page.
2. Social preview: repository **Settings**, **Social preview**, upload
   `web/og.png` (1200 x 630). Links to the repository then show the same card
   as links to the page.
3. Keep releases coming; each release page is indexed separately and links
   back to the site.

## 4. Real links (what moves a competitive term)

One link from a page people actually read is worth more than any number of
directory entries. Post where the readers are, and answer questions there
afterwards. Drafts you can paste and adapt:

**Show HN** (<https://news.ycombinator.com/submit>, title under 80 characters):

> Show HN: Morse code simulator that decodes a PC beeper through the microphone
>
> I wanted to read the Morse a desktop's onboard beeper was keying from a push
> button, so this page listens on the laptop microphone, measures the tone
> power every 10 ms with a Goertzel filter, tracks the noise floor and turns
> marks and gaps into letters, adapting to any speed from 2 to 40 WPM. It grew
> an encoder with a keying guide and Farnsworth spacing, a straight key, iambic
> keyer and bug on the keyboard, practice drills that grade your copy, and a
> chart that lights up as you key. Everything runs in the browser, nothing is
> uploaded; there is also an offline Windows build. The interesting part was
> making the decoder survive room reverb and Windows microphone processing.
> Source: github.com/jetzhu/MorseCodeAudioCoder

**r/amateurradio, r/morse** (also fine for a club mailing list):

> I built a free browser Morse trainer and decoder: it reads a CW tone from the
> microphone (I use a PC beeper, but a practice oscillator or a rig's sidetone
> works), translates text to Morse and plays it with Farnsworth spacing, turns
> the keyboard into a straight key, iambic A/B keyer or bug, grades practice
> drills, and shows a chart that lights up as you key. Nothing is uploaded.
> Feedback on the decoder's behaviour with real fists is very welcome.
> https://jetzhu.github.io/MorseCodeAudioCoder/

**AlternativeTo** (<https://alternativeto.net/manage/add-application/>), as an
alternative to Morse Code World and Morse Code Translator:

> Beeper Morse Console: an online Morse code simulator, decoder, translator
> and trainer. Decodes Morse from the microphone (PC beeper, practice
> oscillator, radio sidetone), plays text as Morse, keys from the keyboard,
> grades practice, shows a chart. Free, open source, runs in the browser
> without uploading audio; offline Windows app available.

Other places that fit: the Hackaday tip line (the beeper angle), the
awesome-ham-radio lists on GitHub (open a pull request adding the link), QRZ
and eHam forums, and your radio club's newsletter.

## 4b. Stars from the page itself

The page and the desktop app carry a quiet "Star on GitHub" link (footer,
download panel, status bar) and ask once, after the first decoded word or the
first correct practice target. Nothing pushier: no popups, no repeats, no
third-party button scripts.

## 5. Watch and adjust

After a few weeks, Search Console's Performance report shows impressions per
query. Expect the page to appear for its own name within days and for generic
terms such as "Morse code simulator" over weeks to months, once a few links
point at it. If a query brings impressions but few clicks, rewrite the
description around that wording.
