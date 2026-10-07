# 02 — Human spacing / review literature for P4 ("Does spaced review help an LM retain facts?")

## TL;DR

- **Spaced beats massed, robustly. Expanding vs. uniform shows no reliable difference.** The meta-analytic estimate is g = 0.034 (Latimier et al. 2021). Total ("absolute") spacing matters a lot, but schedule shape ("relative" spacing) does not (Karpicke & Bauernschmidt 2011).
- **Karpicke & Roediger (2007)** found equal spacing better at 2 days only because the equal schedule delayed the first test. With the first test matched (Exp. 3), there was no difference. Our design matches Exp. 3, so our null is what the human literature predicts.
- **Landauer & Bjork (1978)** found expanding better only for *test-type* practice. For simple re-presentation (restudy), uniform was slightly better (.62 vs .58; checked in the original chapter). LM review is restudy.
- **Expanding gives a during-training advantage** even when final retention ties (Kang et al. 2014). Our "trajectory" result is this known pattern, not an artifact.
- **Degree of learning shifts the forgetting curve's intercept, not its slope** (Slamecka & McElree 1983; Rivera-Lares et al. 2022). That conclusion depends on the measurement scale (Loftus 1985), which is the caveat our "starting point, not forgetting rate" claim needs.
- **Claims to change:**
  - say "consistent with", not "reproduces";
  - drop "spacing effect" unless a massed arm is added;
  - name the theories correctly.

  Section 8.3 has suggested rewrites for each claim. Definitions of all the constructs, with sources, are in Section 1.


Prepared 2026-10-02 for the P4 whitepaper (`P4_whitepaper_writeup.md`, read-only). Scope: the human cognitive-psychology and learning-science literature on spacing, expanding vs. uniform (equal-interval) schedules, theories of spacing, degree of learning vs. rate of forgetting, testing vs. restudy, and computational scheduling models.

**Verification legend** (applies to every entry):
- **[V-full]**: I read the primary full text (publisher or author PDF) and checked the numbers quoted.
- **[V-abs]**: I read the primary abstract (PubMed / publisher / ACL / Semantic Scholar record) and checked the metadata against Crossref. Details beyond the abstract are marked.
- **[V-meta]**: I checked the bibliographic metadata (Crossref / publisher / PubMed). The content summary relies on reputable secondary descriptions, named in each case.
- **UNVERIFIED**: I could not confirm it against a primary source. Do not cite without checking.

DOIs are given as `https://doi.org/...` links. If no DOI exists, I give a publisher, PubMed or author-hosted URL.

---

## 0. Facts about the P4 experiment that matter for the human comparison (from the public repo)

I read the public code (`github.com/anshulmago1/olmo-fictionalqa-review`, `configs/final.yaml`, `src/olmo_core/review_lab/schedules.py`, `trainer.py`, `fictionalqa.py`, at HEAD on 2026-10-02). Several design facts change how the human literature should be applied:

1. **Endpoints are matched.** Both review arms place the first review at Stage-2 step 8 and the last at step 165 (`first_step: 8`, `last_step: 165`, `events: 12`, `expansion_ratio: 1.35`). Reconstructed from the code:
   - Uniform: steps 8, 22, 37, 51, 65, 79, 94, 108, 122, 136, 151, 165.
   - Expanding: steps 8, 10, 13, 17, 22, 29, 38, 51, 68, 91, 123, 165.

   The *first-review delay* and the *retention interval after the last review* are therefore identical. Karpicke & Roediger (2007) identify the first-review delay as the main confound behind expanding-vs-equal differences, so here it is controlled by design.
2. **A "review event" is one optimizer step whose 16-sequence batch is drawn at random, with replacement, from one of the 5 old "skills" (FictionalQA document styles), cycled in turn.** It is not a scheduled repetition of a specific fact.
   - My reconstruction of the data split found about 354 canonical old facts (approximate: I retrieved 7,300 of 7,500 rows).
   - 12 × 16 = 192 review draws therefore give about 0.5 review exposures per old fact. About 59% of old facts are never reviewed in either arm, and most of the rest are reviewed once.
   - At the level of the individual item (the unit in every human study below), neither arm implements an expanding or a uniform multi-review schedule. The schedule exists only at the level of the old-data pool, and at the style level each style gets 2–3 reviews.
   - This is the single biggest disanalogy with the human literature, and the paper should state it.
3. **Review is restudy-like.** It is teacher-forced next-token training on the declarative `fict` statement. Evaluation uses paraphrased questions. There is no retrieval attempt and no possibility of retrieval failure (see §5).
4. **Old-fact loss falls during Stage 2 in all arms, including no-review** ("Old-loss change" −0.223 for no review). Much of the measured "retention" signal therefore reflects shared format/style learning rather than item-specific memory. Human forgetting curves assume that performance on old items does not improve without re-exposure.

---

## 1. Definitions (constructs the paper relies on)

| Construct | Definition (quote or close paraphrase) | Source & location | Status |
|---|---|---|---|
| **Distributed-practice effect** | "The distributed practice effect refers to an effect of interstudy interval (ISI) upon learning, as measured on subsequent tests." | Cepeda et al. (2006), p. 354, "Terminology" | [V-full] |
| **Spacing effect** | "The term spacing effect refers to enhanced learning during spaced as compared with massed study episodes for a given item." Dunlosky et al. (2013, p. 36) use *distributed practice* to cover both spacing effects ("the advantage of spaced over massed practice") and lag effects. | Cepeda et al. (2006), p. 354; Dunlosky et al. (2013), p. 36, §9 | [V-full] |
| **Massed vs. spaced/distributed** | Massed: presentations of an item "separated by 0 items and a time lag of less than 1 s". Spaced/distributed: "a measurable time lag (1 s or longer) separates study episodes for a given item". Carpenter et al. (2012): presentations with a zero gap "are said to be massed"; with a gap greater than zero they "are said to be spaced or distributed". | Cepeda et al. (2006), p. 354; Carpenter et al. (2012), p. 369–370 (ERIC ms. p. 2) | [V-full] |
| **Lag effect** | "the term lag effect refers to comparisons of different levels of spacing, either differing numbers of items … or differing amounts of time". Dunlosky: "the advantage of spacing with longer lags over spacing with shorter lags". | Cepeda et al. (2006), pp. 354–355; Dunlosky et al. (2013), p. 36 | [V-full] |
| **Inter-study interval (ISI) / gap** | "ISI is the interval separating different study episodes of the same materials." Carpenter et al. call it the "spacing gap". | Cepeda et al. (2006), p. 354; Carpenter et al. (2012), p. 370 | [V-full] |
| **Retention interval (RI) / test delay** | The interval "separating the final study episode and a later test". "In studies with more than two study episodes, retention interval still refers to the interval between the last of these study episodes and the final test." Carpenter: "test delay … the time elapsed between the final study presentation and the test." | Cepeda et al. (2006), p. 354; Carpenter et al. (2012), p. 370 | [V-full] |
| **Expanding / equal (fixed, uniform) / contracting schedules** | With three or more episodes, "the ISIs may be equal (fixed), progressively longer (expanding), or progressively shorter (contracting)." Latimier et al.: in the uniform schedule "spacing intervals are kept constant", while in the expanding schedule "spacing intervals increase after every re-exposure"; the contracting schedule decreases them. | Cepeda et al. (2006), p. 355; Latimier et al. (2021), p. 961 | [V-full] |
| **Expanding retrieval practice** | "attempting to retrieve an item immediately after it has been studied (an immediate first test) and then gradually increasing the spacing interval between successive retrieval attempts … intended to ensure a high level of retrieval success on the first test and to increase the difficulty of retrieval attempts on subsequent repeated tests." | Karpicke & Roediger (2007), p. 704 | [V-full] |
| **Relative vs. absolute spacing** | Relative spacing is "how the repeated episodes are spaced relative to one another". Absolute spacing is the total spacing across repeated tests. | Latimier et al. (2021), p. 961, citing Wiseheart et al. (2019, p. 555); Karpicke & Bauernschmidt (2011), abstract | [V-full]/[V-abs] |
| **Retrieval practice; testing effect** | "Taking a memory test not only assesses what one knows, but also enhances later retention, a phenomenon known as the testing effect." Retrieval practice "refers to any activity that requires the learner to retrieve previously learnt information from memory." | Roediger & Karpicke (2006), p. 249 (abstract); Latimier et al. (2021), p. 959 | [V-full] |
| **Restudy vs. test-type rehearsal** | Landauer & Bjork distinguish two kinds of practice. In one, "new information is presented repeatedly for study". In the other, "a fact is presented just once, and subsequent rehearsal takes the form of 'tests'". | Landauer & Bjork (1978), p. 625 | [V-full] |
| **Review / relearning** | Dunlosky: distributed practice is "a particular schedule of learning episodes, as opposed to a particular kind of learning episode … [which] could involve restudying material, retrieving information from memory, or practicing skills." "Relearning" (Rawson & Dunlosky, 2011) means further practice sessions after an initial criterion is reached. | Dunlosky et al. (2013), p. 36, §9.2a; Rawson & Dunlosky (2011), abstract | [V-full]/[V-abs] |
| **Desirable difficulties** | Conditions that "trigger encoding and retrieval processes that support learning, comprehension, and remembering". They include spacing, interleaving, varying conditions and "using tests, rather than presentations, as study events". A difficulty becomes undesirable if "the learner does not have the background knowledge or skills to respond to them successfully". | Bjork & Bjork (2011/2014), §"Introducing Desirable Difficulties" (2014 ed. pp. 59–68); term coined in Bjork (1994) | [V-full] (2014 ed.) / [V-meta] (1994) |
| **Storage vs. retrieval strength (New Theory of Disuse)** | "Storage strength reflects how entrenched or interassociated a memory representation is … retrieval strength reflects the current activation or accessibility … storage strength acts to retard the loss (forgetting) and enhance the gain (relearning) of retrieval strength." Also: "when some skill or knowledge is maximally accessible from memory, little or no learning results from additional instruction or practice." | Bjork & Bjork (2014 ed.), "Storage Strength Versus Retrieval Strength" and "Interleaving" sections | [V-full] |
| **Forgetting curve / savings** | Ebbinghaus measured retention as *savings*: the proportion of original learning work saved when relearning after a delay (e.g., 58% saved after about 19 min, Ch. 7). Wixted (2004) reviews the shape of forgetting (approximately power-like, with an ever-decreasing proportional rate). | Ebbinghaus (1885/1913), Ch. 7; Wixted (2004) | [V-full] (Ebbinghaus) / [V-meta] (Wixted) |
| **Leitner system** | A flashcard box system: cards answered correctly move to a less frequently reviewed box; errors return cards to box 1. Introduced in Leitner's popular book *So lernt man lernen*. Modelled formally by Reddy et al. (2016) as a queueing network. | Leitner (1972); Reddy et al. (2016) | Publisher (Herder, Freiburg) confirmed via Open Library. **The 1972 first-edition date is UNVERIFIED against a primary catalogue** (Open Library lists editions from 1987 on). |
| **SM-2 / SuperMemo** | "I(1):=1, I(2):=6, for n>2: I(n):=I(n-1)*EF". EF is an "easiness factor" that starts at 2.5, has a floor of 1.3 and is updated by EF' := EF + (0.1 − (5−q)(0.08 + (5−q)·0.02)). Intervals are in days. Anki's default scheduler descends from SM-2. | Wozniak (1990/1998), SuperMemo web page | [V-full] (web page). The Anki lineage is common knowledge, not verified here. |
| **Pimsleur graduated-interval recall** | An expanding "memory schedule" for language learning. | Pimsleur (1967), *Modern Language Journal* 51(2), 73–75 | [V-meta]. Its specific interval values are **UNVERIFIED** (not checked against the article). |

---

## 2. The spacing / distributed-practice effect

**Ebbinghaus (1885/1913).** Ebbinghaus, H. (1913). *Memory: A contribution to experimental psychology* (H. A. Ruger & C. E. Bussenius, Trans.). Teachers College, Columbia University. (Original work published 1885.) https://psychclassics.yorku.ca/Ebbinghaus/index.htm. [V-full, Ch. 7–8]
- **Finding.** Ch. 8 (§34): "38 repetitions, distributed in a certain way over the three preceding days, had just as favorable an effect as 68 repetitions made on the day just previous." He concludes that "a suitable distribution of them over a space of time is decidedly more advantageous than the massing of them at a single time." Ch. 7 gives the savings-based forgetting curve. A modern single-subject replication is Murre & Dros (2015), *PLOS ONE*, 10(7), e0120644, https://doi.org/10.1371/journal.pone.0120644 [V-meta].
- **Use.** Historical anchor. Savings, the speed of relearning, is a natural LM metric that the paper does not report (see Walsh et al., 2018, §7).

**Cepeda et al. (2006).** Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin, 132*(3), 354–380. https://doi.org/10.1037/0033-2909.132.3.354. [V-full]
- **Finding.** The review covers 839 assessments in 317 experiments from 184 articles. The ISI producing maximal retention increased as the RI increased. 80% of comparisons used RIs under 1 day.
- **On expanding vs. fixed (p. 364, Table 8):** 22 comparisons from 18 studies. Expanding averaged 62.0% and fixed 58.6%, t(42) = 0.5, p = .61, with large between-study variability, so the authors call conclusions "necessarily tentative." Their conclusion (p. 366): expanding intervals "either benefit learning or produce effects similar to studying with fixed spacing."
- **Theories (pp. 368–370):** They review deficient processing, encoding variability, consolidation and study-phase retrieval. Deficient processing "does not readily survive", and only encoding variability had been shown by modelling to produce an optimal ISI that increases with RI.
- (Dunlosky et al. 2013 summarise this review as "254 studies involving more than 14,000 participants … 47% vs 37%". That is a different count from Cepeda's own abstract, so cite Cepeda's numbers.)
- **Use.** This is the correct citation for the robustness of spacing and for the ISI × RI interaction. **It is not support for "expanding schedules greatly improve memory".** Cepeda's own expanding-vs-fixed comparison was non-significant and called tentative.

**Cepeda et al. (2008).** Cepeda, N. J., Vul, E., Rohrer, D., Wixted, J. T., & Pashler, H. (2008). Spacing effects in learning: A temporal ridgeline of optimal retention. *Psychological Science, 19*(11), 1095–1102. https://doi.org/10.1111/j.1467-9280.2008.02209.x. [V-abs]
- **Finding.** More than 1,350 participants, gaps up to 3.5 months, RIs up to 1 year. At each RI, performance first rose and then gently declined as the gap grew. The optimal gap increased with RI but fell as a proportion of it: "from about 20 to 40% of a 1-week test delay to about 5 to 10% of a 1-year test delay."
- **Use.** The optimal schedule depends on the retention interval. P4 tests one RI (about 194 steps after the last review, about 375 after Stage 1), so it cannot speak to RI-dependence. A natural follow-up is to vary buffer length or the absolute gap.

**Cepeda et al. (2009).** Cepeda, N. J., Coburn, N., Rohrer, D., Wixted, J. T., Mozer, M. C., & Pashler, H. (2009). Optimizing distributed practice: Theoretical analysis and practical implications. *Experimental Psychology, 56*(4), 236–246. https://doi.org/10.1027/1618-3169.56.4.236. [V-abs]
- **Finding.** "An optimal gap improved final recall by up to 150%." Gap effects were non-monotonic, with test delays up to 6 months.
- **Use.** Supports the claim that *absolute* gap matters a great deal, in contrast to relative schedule shape.

**Dunlosky et al. (2013).** Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham, D. T. (2013). Improving students' learning with effective learning techniques: Promising directions from cognitive and educational psychology. *Psychological Science in the Public Interest, 14*(1), 4–58. https://doi.org/10.1177/1529100612453266. [V-full]
- **Finding.** Distributed practice and practice testing were the only two techniques rated "high utility". §9.1 summarises deficient-processing, reminding and consolidation accounts. Bahrick (1979) is the illustrative case: the 30-day-gap condition had the most forgetting between sessions but the best retention at 30 days.
- **Use.** An authoritative definition source. Bahrick's pattern, where worse performance during training goes with better long-term retention, is a useful contrast to P4's "trajectory" metric.

**Carpenter et al. (2012).** Carpenter, S. K., Cepeda, N. J., Rohrer, D., Kang, S. H. K., & Pashler, H. (2012). Using spacing to enhance diverse forms of learning: Review of recent research and implications for instruction. *Educational Psychology Review, 24*(3), 369–378. https://doi.org/10.1007/s10648-012-9205-z. [V-full, ERIC manuscript ED536925]
- **Finding.** On fixed vs. expanding schedules: "Experiments comparing fixed and expanding schedules have produced equivocal results." Also: "Any form of spacing—whether it is fixed or expanding—appears to promote learning." Practical advice: "students and teachers do not need to be overly concerned about whether the spacing gaps that separate repeated study sessions are equal or not."
- **Use.** A near-verbatim human precedent for P4's practical conclusion. Cite it as the prior consensus rather than claiming novelty.

**Kang (2016).** Kang, S. H. K. (2016). Spaced repetition promotes efficient and effective learning: Policy implications for instruction. *Policy Insights from the Behavioral and Brain Sciences, 3*(1), 12–19. https://doi.org/10.1177/2372732215624708. [V-meta; abstract not retrievable (SAGE 403)]
- **Finding (secondary).** A policy review of spaced repetition. Per secondary quotations, it notes that whether expanding beats equal spacing "probably depends on factors such as the difficulty of the to-be-learned material" and highlights Kang et al. (2014).
- **Use.** A short policy citation only. Check the text before quoting.

**Donovan & Radosevich (1999).** Donovan, J. J., & Radosevich, D. J. (1999). A meta-analytic review of the distribution of practice effect: Now you see it, now you don't. *Journal of Applied Psychology, 84*(5), 795–805. https://doi.org/10.1037/0021-9010.84.5.795. [V-meta; abstract via secondary sources]
- **Finding.** 63 studies and 112 effect sizes, with mean d = 0.46. Effects were smaller for more complex tasks and in more rigorous studies. Most tasks were motor tasks.
- **Use.** Spacing effects shrink for complex tasks, a caution for generalising to LM knowledge.

**Wiseheart et al. (2019).** Wiseheart, M., Küpper-Tetzel, C. E., Weston, T., Kim, A. S. N., Kapler, I. V., & Foot-Seymour, V. (2019). Enhancing the quality of student learning using distributed practice. In J. Dunlosky & K. A. Rawson (Eds.), *The Cambridge handbook of cognition and education* (pp. 550–584). Cambridge University Press. https://doi.org/10.1017/9781108235631.023. [V-meta]
- **Finding (via Latimier et al., 2021, p. 960).** Pooling Cepeda (2006) and Moss (1996) gives d = 0.85 for spaced verbal learning. p. 555 defines relative spacing.
- **Use.** A broad review. Cite for the definition of relative spacing.

**Recent reviews and meta-analyses (2019–2026)**
- **Kim et al. (2019).** Kim, A. S. N., Wong-Kee-You, A. M. B., Wiseheart, M., & Rosenbaum, R. S. (2019). The spacing effect stands up to big data. *Behavior Research Methods, 51*(4), 1485–1497. https://doi.org/10.3758/s13428-018-1184-7. [V-full abstract]
  - **Finding.** In workplace training data from 10,514 people, "the optimal amount of spacing between repeated retrieval events increased as the retention interval increased."
  - **Use.** Evidence that the ridgeline replicates at scale.
- **Kim & Webb (2022).** Kim, S. K., & Webb, S. (2022). The effects of spaced practice on second language learning: A meta-analysis. *Language Learning, 72*(1), 269–319. https://doi.org/10.1111/lang.12479. [V-meta; abstract content via secondary sources]
  - **Finding.** 98 effect sizes. Shorter spacing was as effective as longer spacing on immediate posttests but less effective on delayed ones.
- **Mawson & Kang (2025).** Mawson, R. D., & Kang, S. H. K. (2025). The distributed practice effect on classroom learning: A meta-analytic review of applied research. *Behavioral Sciences, 15*(6), 771. https://doi.org/10.3390/bs15060771. [V-abs]
  - **Finding.** 22 reports and 31 effect sizes, d = 0.54 [0.31, 0.77]. Effects were larger with longer RIs and fewer re-exposures.
- **Murray et al. (2025).** Murray, E., Horner, A. J., & Göbel, S. M. (2025). A meta-analytic review of the effectiveness of spacing and retrieval practice for mathematics learning. *Educational Psychology Review, 37*, Article 75. https://doi.org/10.1007/s10648-025-10035-1. [V-full abstract]
  - **Finding.** Spaced vs. massed g = 0.28 (27 studies), isolated g = 0.43 vs. course-embedded g = 0.24. Testing vs. restudy g = 0.18 with a CI crossing 0.
  - **Use.** Spacing benefits are domain-dependent and smaller for complex material.
- **Cowan et al. (2024).** Cowan, E. T., Zhang, Y., Rottman, B. M., & Murty, V. P. (2024). The effects of mnemonic variability and spacing on memory over multiple timescales. *PNAS, 121*(12), e2311077121. https://doi.org/10.1073/pnas.2311077121. [V-meta + publisher summary]
  - **Finding.** For item memory, both variability and spacing helped. For associative memory, spacing benefits appeared only without variability.
  - **Use.** A recent test of encoding variability (§4).

---

## 3. Expanding vs. equal-interval (uniform) schedules: the core comparison

### 3.1 Summary table

RI = retention interval after the last practice event.

| Study | Practice type | Schedules (gaps) | Short-RI result | Long-RI result | First interval / retrieval success |
|---|---|---|---|---|---|
| Landauer & Bjork 1978 | Tests (names) **and** repetitions (restudy) | Exp. II: 0-1-3-8 vs 3-3-3-3 | 30 min RI. Tests: expanding .66 vs uniform .56 (p<.01). **Repetitions: expanding .58 vs uniform .62 (n.s.)**. Interaction significant. | not tested | Expanding works "because it keeps the probability of a successful test relatively high" |
| Cull 2000 | Tests and study, within and across days | massed / uniform / expanding | — | Large spacing effects. Expanding "did not independently benefit learning" in any situation | — |
| Balota et al. 2006 | Tests (young, old, AD) | expanding vs equal | Expanding better *during acquisition* | Advantage lost at final recall (Exp 1). No difference with feedback (Exp 2–3) | — |
| Karpicke & Roediger 2007 | Tests (GRE pairs), ± feedback | 1-5-9 vs 5-5-5 (trials) | 10 min: expanding .71 vs equal .62 (Exp 1) | **2 days: equal .45 vs expanding .33 (Exp 1); .60 vs .49 with feedback (Exp 2)** | Exp 3, first test equated: 2 days .43 vs .45 (both immediate first test), .51 vs .52 (both delayed). Delaying the first test is the key factor |
| Logan & Balota 2008 | Tests (young/old) | 1-2-3/2-2-2; 1-3-5/3-3-3; 1-3-8/4-4-4 | Same day: expanding effect only in older adults | 24 h: no expanding advantage. Young adults showed an expanding *disadvantage* | Expanding gave higher learning-phase success |
| Karpicke & Roediger 2010 | Recall of texts | expanding vs equal | — | 1 week: no difference | — |
| Storm, Bjork & Storm 2010 | Tests of text, no feedback | immediate-first expanding vs uniform | — | 1 week: expanding helped **only when the intervening task was highly interfering** (Exp 2: .43 vs .19) | Benefit "depends on the degree to which the to-be-learned information is vulnerable to forgetting" |
| Maddox et al. 2011 | Tests (young/old), 60 min RI | 0-2-4-6-8 vs 0-5-5-5-5 (Exp 1). 0-1-6-8-10 vs 5-5-5-5-5 (Exp 2) | Exp 1 n.s. Exp 2 expanding about +.10 | — | Benefit tied to preventing forgetting before early retrievals |
| Karpicke & Bauernschmidt 2011 | Tests after criterion learning | expanding / equal / contracting at several absolute spacings | — | **Absolute spacing gave about a 200% gain. No relative-schedule effect** | Rising retrieval difficulty under expanding did not translate into gains |
| Gerbier & Koenig 2012 | **Restudy over days** | days 1-2-3-7 (expanding), 1-3-5-7 (uniform), 1-5-6-7 (contracting) | — | Day 9: expanding best (Exp 1, web). Expanding = uniform best (Exp 2, lab) | Recognition at repetition predicted later recall (study-phase retrieval) |
| Kang, Lindsey, Mozer & Pashler 2014 | Tests + feedback over 4 weeks | expanding (days 3, 9, 28) vs equal (days 10, 19, 28) | — | Day 84 (56 days after training): .49 vs .46, n.s. **Less forgetting under expanding (.13 vs .19, d = 0.38)**. **Average recallability during training .49 vs .41 (p<.001)** | Same last session (day 28) |
| Küpper-Tetzel, Kapler & Wiseheart 2014 | **Restudy** (3 sessions) | contracting / equal / expanding | Contracting best up to 7-day RI | **35-day RI: equal and expanding better than contracting** | Fits contextual-variability theory |
| Gerbier, Toppino & Koenig 2015 | **Restudy over 13 days** | days 1-2-13 vs 1-7-13 vs 1-12-13 | — | Expanding generally best. Advantage may grow with RI (2, 6, 13 days) | Interpreted via study-phase retrieval |
| Mettler, Massey & Kellman 2016 | Adaptive vs fixed | adaptive (ARTS) vs expanding vs equal | — | Adaptive beat fixed at immediate and delayed tests. **"No evidence … for differences between expanding and equal spacing"** | — |
| Toppino, Phelan & Gerbier 2018 | **Restudy** across 13 days | expanding / contracting / uniform | — | 2-week RI: expanding best after *low* initial training; no advantage after *high* initial training | Study-phase retrieval |
| Yan, Eglington & Garcia 2020 | Tests + feedback, one session | expanding vs uniform, fixed time | — | Delayed test: critical items equal (Exp 1) or better (Exp 2). Expanding is more *efficient*: 71% and 41% more words recalled when extra items are inserted | — |
| **Latimier, Peyre & Ramus 2021 (meta-analysis)** | Retrieval practice only | 16 studies, 54 effect sizes | — | **g = 0.034 [−0.10, 0.17], p = .59, I² = 0%. RI did *not* moderate (β = 0.03 [−0.07, 0.13])** | >4 exposures: g = 0.20 [−0.07, 0.47] vs ≤4: g = −0.04 (moderator p = .09). First-retrieval placement n.s. (β = 0.25, p = .35) |

### 3.2 Entries

**Landauer & Bjork (1978).** Landauer, T. K., & Bjork, R. A. (1978). Optimum rehearsal patterns and name learning. In M. M. Gruneberg, P. E. Morris, & R. N. Sykes (Eds.), *Practical aspects of memory* (pp. 625–632). Academic Press. Author-hosted PDF: https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/Landauer.Bjork_.1978.pdf. [V-full]
- **Finding.** The abstract says: "a pattern of increasing intervals between successive rehearsals was best for test-type practice, while uniform spacing was slightly better if the information was repeated."
  - Exp. I (N = 468, RI 30 min): expanding beat comparable uniform patterns, z = 2.6, p < .01.
  - Exp. II (N = 218): for tests, expanding .66 vs uniform .56 (t = 3.16). For repetitions, expanding .58 vs uniform .62 (n.s.). The schedule × practice-type interaction was significant.
  - The authors predicted in advance that "uniform spacing should be better for repetition-type practice."
- **Use.** This is the most important citation for P4. The original expanding-schedule paper (a) tested only a 30-minute RI and (b) predicted and found that for *restudy* (re-presentation, which is what LM gradient review is), uniform is at least as good. The paper currently cites L&B 1978 for expanding schedules "shown to greatly improve memory". That overstates the effect (about 10 percentage points, short delay, test-type practice only) and omits the restudy result that directly predicts P4's null.

**Cull (2000).** Cull, W. L. (2000). Untangling the benefits of multiple study opportunities and repeated testing for cued recall. *Applied Cognitive Psychology, 14*(3), 215–235. https://doi.org/10.1002/(SICI)1099-0720(200005/06)14:3%3C215::AID-ACP640%3E3.0.CO;2-1. [V-meta; abstract content via Wiley abstract as quoted in search and Cepeda 2006]
- **Finding.** Four experiments spanning single sessions to reviews days later. Distributed practice had large effects, but "expanding test spacing … did not independently benefit learning in any of the learning situations studied." Cepeda et al. (2006, Table 9) list Cull (2000) as favouring fixed intervals at ISIs and RIs of 1 day or more.
- **Use.** Early multi-day evidence against an expanding advantage.

**Balota, Duchek & Logan (2007).** Balota, D. A., Duchek, J. M., & Logan, J. M. (2007). Is expanded retrieval practice a superior form of spaced retrieval? A critical review of the extant literature. In J. S. Nairne (Ed.), *The foundations of remembering: Essays in honor of Henry L. Roediger III* (pp. 83–105). Psychology Press. URL: http://psychnet.wustl.edu/coglab/wp-content/uploads/2015/01/2007-Is-expanded.pdf. [V-meta. The URL appeared in search results, but my direct download returned an HTML page, so the link may have moved. The content summary comes from secondary sources and the opening text shown in search.]
- **Finding.** A qualitative review: equal-interval spacing reliably beats massed practice, while evidence for an expanded-over-equal advantage is weak and inconsistent. It notes the classic spacing × RI interaction: massed is better at short RIs and spaced at long RIs.
- **Use.** The standard review to cite alongside Latimier.

**Balota et al. (2006).** Balota, D. A., Duchek, J. M., Sergent-Marshall, S. D., & Roediger, H. L., III. (2006). Does expanded retrieval produce benefits over equal-interval spacing? Explorations of spacing effects in healthy aging and early stage Alzheimer's disease. *Psychology and Aging, 21*(1), 19–31. https://doi.org/10.1037/0882-7974.21.1.19. [V-abs]
- **Finding.** Expanded retrieval beat equal during acquisition, but "this benefit was lost in final cued recall". With corrective feedback there was "no evidence of a difference".
- **Use.** A human analogue of P4's pattern: expanding looks better during training, equal at the end.

**Logan & Balota (2008).** Logan, J. M., & Balota, D. A. (2008). Expanded vs. equal interval spaced retrieval practice: Exploring different schedules of spacing and retention interval in younger and older adults. *Aging, Neuropsychology, and Cognition, 15*(3), 257–280. https://doi.org/10.1080/13825580701322171. [V-abs]
- **Finding.** Schedules were matched on average spacing (1-2-3/2-2-2; 1-3-5/3-3-3; 1-3-8/4-4-4). Expanded items had higher success during learning. Same day, only older adults showed an expansion effect. After 24 h, "the final recall advantage for items in the expanded condition was lost in both groups, and … at a significant recall disadvantage for younger adults."
- **Use.** Supports "no long-delay advantage" for expanding, and matched average spacing is the right control. P4 matches the count and endpoints but not the mean gap; the expanding arm's mean gap is the same by construction because both endpoints are shared.

**Karpicke & Roediger (2007).** Karpicke, J. D., & Roediger, H. L., III. (2007). Expanding retrieval practice promotes short-term retention, but equally spaced retrieval enhances long-term retention. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 33*(4), 704–719. https://doi.org/10.1037/0278-7393.33.4.704. [V-full]
- **Finding.** GRE word pairs; spacing counted in intervening trials (1-5-9 vs 5-5-5, equal total).
  - Exp 1: at 10 min, expanding .71 vs equal .62; at 2 days, equal .45 vs expanding .33.
  - Exp 2, with feedback: at 2 days, .60 vs .49.
  - Exp 3 crossed first-test delay with schedule. At 2 days the results were .43 (0-1-5-9) vs .45 (0-5-5-5), and .51 (5-1-5-9) vs .52 (5-5-5-5). "Delaying the first test improved long-term retention, regardless of how the repeated tests were spaced."
  - Conclusion: "Expanding the interval between repeated tests had little effect on long-term retention in 3 experiments."
- **Use.**
  - Their headline *reversal* (equal > expanding at a delay) arises because the equal schedule's *first* test is delayed.
  - P4 equates the first review (step 8 in both arms) and the last review (step 165). It therefore corresponds to K&R's Exp 3, where the schedule shape had no effect. P4's null is the result Exp 3 predicts.
  - The paper should cite Exp 3 specifically and not say equal "beats" expanding.
  - The "long delay" in K&R is 2 days, and their intervals are a handful of intervening trials.

**Karpicke & Roediger (2010).** Karpicke, J. D., & Roediger, H. L., III. (2010). Is expanding retrieval a superior method for learning text materials? *Memory & Cognition, 38*(1), 116–124. https://doi.org/10.3758/MC.38.1.116. [V-abs]
- **Finding.** Text passages, 1-week final test. Repeated tests beat a single test, and feedback helped, but "the spacing of the tests did not matter".
- **Use.** Extends the null from word pairs to prose, which is closer to FictionalQA statements.

**Storm, Bjork & Storm (2010).** Storm, B. C., Bjork, R. A., & Storm, J. C. (2010). Optimizing retrieval as a learning event: When and why expanding retrieval practice enhances long-term retention. *Memory & Cognition, 38*(2), 244–253. https://doi.org/10.3758/MC.38.2.244. [V-full, partial read]
- **Finding.** Educational passages, repeated tests without feedback, 1-week test. Expanding helped "only when the task between successive retrievals was highly interfering with memory for the passage". In Exp 2 (interfering), expanding .43 vs uniform .19. Under low interference (Exp 1) the uniform condition was numerically higher and n.s.
- **Use.** Continued training on *new FictionalQA facts* is a high-interference regime, so this study predicts an expanding benefit. The benefit in Storm et al. works by preserving *retrieval success* on early tests without feedback, a mechanism absent in gradient restudy, which always shows the target. This explains why the interference prediction need not transfer.

**Maddox, Balota, Coane & Duchek (2011).** Maddox, G. B., Balota, D. A., Coane, J. H., & Duchek, J. M. (2011). The role of forgetting rate in producing a benefit of expanded over equal spaced retrieval in young and older adults. *Psychology and Aging, 26*(3), 661–670. https://doi.org/10.1037/a0022942. [V-abs/PMC summary]
- **Finding.** Exp 1 (0-2-4-6-8 vs 0-5-5-5-5): no difference. Exp 2 (0-1-6-8-10 vs 5-5-5-5-5): expanding about +.10 in both age groups. The benefit depends on whether early retrievals succeed, and success at attempt 2 strongly predicted final recall.
- **Use.** Expanding helps when it rescues early retrieval from fast forgetting. This is not applicable to restudy.

**Karpicke & Bauernschmidt (2011).** Karpicke, J. D., & Bauernschmidt, A. (2011). Spaced retrieval: Absolute spacing enhances learning regardless of relative spacing. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 37*(5), 1250–1257. https://doi.org/10.1037/a0023436. [V-abs]
- **Finding.** After criterion learning, three repeated tests. Long absolute spacing gave a "200% improvement in long-term retention relative to … no spacing", but "no evidence that a particular relative spacing schedule (expanding, equal, or contracting) was inherently superior."
- **Use.** The cleanest human framing for P4. P4 manipulated *relative* spacing with absolute span fixed (steps 8–165 in both arms), so a null is the human-predicted outcome. P4 did not manipulate absolute spacing (e.g., a massed review block), so it cannot say spacing "does not matter".

**Kang, Lindsey, Mozer & Pashler (2014).** Kang, S. H. K., Lindsey, R. V., Mozer, M. C., & Pashler, H. (2014). Retrieval practice over the long term: Should spacing be expanding or equal-interval? *Psychonomic Bulletin & Review, 21*(6), 1544–1550. https://doi.org/10.3758/s13423-014-0636-z. [V-full]
- **Finding.** 60 Japanese–English pairs with feedback. Expanding sessions on days 3, 9, 28 vs equal on days 10, 19, 28; final test on day 84.
  - Final test: .49 vs .46, n.s.
  - Forgetting after training: expanding .13 vs equal .19, p = .026, d = 0.38.
  - Mean interpolated recall during training: .51 vs .43 by sampled curves, and .49 vs .41 by subjects, t(36) = 3.65, p < .001.
  - The authors argue "a more generally important goal would be to maintain high average performance over a considerable period of training."
- **Use.**
  - This is the closest human analogue to P4's whole pattern: same last-review time, equal final retention, and an expanding advantage *during* training.
  - P4 dismisses its trajectory advantage (−0.436 vs −0.423) as an "artifact". Kang et al. treat the equivalent quantity, average accessibility, as a primary outcome. For a deployed, continually training LM that is queried throughout training, this is arguably the relevant metric.
  - The P4 one-pager (`P4_onepager.md`) reports expanding > uniform on AUC in a Pythia-162M, 20-seed experiment. That is consistent with Kang et al. and should be reconciled with the whitepaper's "spacing did not matter" conclusion.

**Latimier, Peyre & Ramus (2021).** Latimier, A., Peyre, H., & Ramus, F. (2021). A meta-analytic review of the benefit of spacing out retrieval practice episodes on retention. *Educational Psychology Review, 33*(3), 959–987. https://doi.org/10.1007/s10648-020-09572-8. [V-full, author PDF]
- **Spaced vs. massed retrieval:** 11 studies, 39 effect sizes. g = 1.01 [0.68, 1.34], or g = 0.74 [0.55, 0.91] after trim-and-fill for publication bias.
- **Expanding vs. uniform:** 16 studies, 54 effect sizes. g = 0.034 [−0.10, 0.17], p = .59, I² = 0%, no publication bias.
- **Moderators:** RI did *not* moderate (β = 0.03 [−0.07, 0.13], p = .53). Number of exposures (>4 vs 4): β = 0.22, p = .09, with g = 0.20 [−0.07, 0.47] for >4 and g = −0.04 for ≤4. Placement of the first retrieval: β = 0.25, p = .35. Feedback: n.s.
- **Design note:** In 34 of 54 comparisons the expanding schedule's first retrieval was immediate and the uniform schedule's was delayed.
- **Conclusion:** the evidence does "not support the widely held belief that inter-retrieval intervals should be progressively increased until a retention test."
- **Scope:** retrieval practice only. They note that Toppino et al. (2018) found an expanding advantage after spaced *reading* but not spaced retrieval, which they could not evaluate.
- **Use.**
  - The primary quantitative citation for "expanding ≈ uniform".
  - Precision matters. The meta-analysis shows *no average difference and no RI moderation*. It does not show that "equal beats expanding at long delays". The P4 abstract's wording ("no reliable long-delay advantage") is acceptable. The conclusion's "matches or beats … once memory is tested after a long delay" leans on K&R (2007) Exp 1–2 and is not the meta-analytic consensus.
  - The exposures moderator (more exposures, trend toward expanding) is relevant because P4 uses 12 review events. Per item, however, P4 gives about 0.5 review exposures (§0), so the moderator does not straightforwardly apply.

**Gerbier & Koenig (2012).** Gerbier, E., & Koenig, O. (2012). Influence of multiple-day temporal distribution of repetitions on memory: A comparison of uniform, expanding, and contracting schedules. *Quarterly Journal of Experimental Psychology, 65*(3), 514–525. https://doi.org/10.1080/17470218.2011.600806. [V-abs]
- **Finding.** Four presentations (restudy) over 7 days, test on day 9. Exp 1 (online): expanding best. Exp 2 (lab): expanding and uniform both best. Items recognised at repetition were recalled better later.
- **Use.** The restudy-over-days literature leans somewhat toward expanding ≥ uniform, unlike retrieval-practice null results.

**Gerbier, Toppino & Koenig (2015).** Gerbier, E., Toppino, T. C., & Koenig, O. (2015). Optimising retention through multiple study opportunities over days: The benefit of an expanding schedule of repetitions. *Memory, 23*(6), 943–954. https://doi.org/10.1080/09658211.2014.944916. [V-abs]
- **Finding.** Vocabulary restudy on days 1-2-13 (expanding), 1-7-13 (uniform) and 1-12-13 (contracting); RIs of 2, 6 and 13 days. "The expanding schedule generally led to better performance", and the benefit "may be greater when the RI is longer."
- **Use.** The strongest human evidence *for* expanding at longer RIs, and it comes from restudy. It is a counterexample to any blanket claim that humans show no long-delay expanding advantage.
- (The task listed "Gerbier & Toppino 2015". There are two different 2015 papers: this *Memory* article, with Koenig, and the review below.)

**Gerbier & Toppino (2015).** Gerbier, E., & Toppino, T. C. (2015). The effect of distributed practice: Neuroscience, cognition, and education. *Trends in Neuroscience and Education, 4*(3), 49–59. https://doi.org/10.1016/j.tine.2015.01.001. [V-meta]
- **Finding.** A review of behavioural, neuroimaging and neurophysiological work on distributed practice and its theoretical accounts.
- **Use.** A review citation for theories and neural mechanisms.

**Toppino, Phelan & Gerbier (2018).** Toppino, T. C., Phelan, H.-A., & Gerbier, E. (2018). Level of initial training moderates the effects of distributing practice over multiple days with expanding, contracting, and uniform schedules: Evidence for study-phase retrieval. *Memory & Cognition, 46*(6), 969–978. https://doi.org/10.3758/s13421-018-0815-7. [V-abs]
- **Finding.** Over 13 days, with a 2-week RI, there was "an expanding-schedule superiority following low-level initial training but not following high-level initial training."
- **Use.** P4's Stage 1 gives about 2.7 exposures per old fact (60 × 16 draws over about 354 facts), with old-fact loss still near 3.8–4.0. That is low initial learning, a regime in which this study predicts an expanding advantage. The prediction may fail because, per item, P4 does not implement an expanding schedule (§0).

**Küpper-Tetzel, Kapler & Wiseheart (2014).** Küpper-Tetzel, C. E., Kapler, I. V., & Wiseheart, M. (2014). Contracting, equal, and expanding learning schedules: The optimal distribution of learning sessions depends on retention interval. *Memory & Cognition, 42*(5), 729–741. https://doi.org/10.3758/s13421-014-0394-1. [V-abs]
- **Finding.** Restudy of pairs over three sessions; RIs of 0, 1, 7 or 35 days. "A contracting learning schedule was beneficial for retention intervals up to 7 days, but both equal and expanding learning schedules were better for a long retention interval of 35 days." The results are best fit by contextual-variability theory.
- **Use.** With restudy at a long RI, equal ≈ expanding, which matches P4. It also shows that schedule effects are RI-dependent, so P4's single-RI result does not license general claims.

**Mettler, Massey & Kellman (2016).** Mettler, E., Massey, C. M., & Kellman, P. J. (2016). A comparison of adaptive and fixed schedules of practice. *Journal of Experimental Psychology: General, 145*(7), 897–917. https://doi.org/10.1037/xge0000170. [V-abs]
- **Finding.** Adaptive (response-time-based) scheduling beat fixed schedules at immediate and delayed tests, and yoked controls show the gain comes from adaptation. "No evidence was found for differences between expanding and equal spacing."
- **Use.** Supports moving from fixed shapes to *state-dependent* (loss-triggered) review. The P4 code already contains an `adaptive_due` condition; human data suggest this is where a gain could come from.

**Yan, Eglington & Garcia (2020).** Yan, V. X., Eglington, L. G., & Garcia, M. A. (2020). Learning better, learning more: The benefits of expanded retrieval practice. *Journal of Applied Research in Memory and Cognition, 9*(2), 204–214. https://doi.org/10.1016/j.jarmac.2020.03.002. [V-abs]
- **Finding.** In a fixed time budget, expanding retrieval with feedback lets extra items be interleaved. Critical items were recalled equally (Exp 1) or better (Exp 2), and expanding yielded 71% and 41% more words recalled overall.
- **Use.** A recent (2020) result favouring expanding on *efficiency* grounds rather than per-item retention. P4 equates compute per arm but does not exploit freed slots.

**Rawson & Dunlosky (2011).** Rawson, K. A., & Dunlosky, J. (2011). Optimizing schedules of retrieval practice for durable and efficient learning: How much is enough? *Journal of Experimental Psychology: General, 140*(3), 283–302. https://doi.org/10.1037/a0023956. [V-abs]
- **Finding.** Initial criterion and relearning were subadditive: criterion effects faded as relearning sessions increased. The recommendation is 3 correct recalls initially and then 3 widely spaced relearning sessions.
- **Use.** Relearning sessions, not schedule shape, drive durability.

---

## 4. Theories of the spacing effect, and machine analogues

| Theory | Core claim | Key sources | Plausible SGD/LM analogue | Prediction for P4 (expanding vs uniform, restudy, matched endpoints) |
|---|---|---|---|---|
| **Deficient processing / inattention** | A repetition soon after the first gets less processing because the item is still fresh and easy. | Hintzman (1974); Cepeda et al. (2006) pp. 368–369, who conclude it "does not readily survive"; Dunlosky et al. (2013) §9.1 | **Strong analogue.** The per-example gradient magnitude scales with current loss (or error), so reviewing an item whose loss is still low yields small updates. This mirrors Bjork & Bjork's "when … maximally accessible … little or no learning results." | Expanding front-loads 6 of 12 reviews into steps 8–29, when old-fact loss is lowest, which predicts slightly *less* efficient review than uniform. P4 saw no difference. |
| **Encoding / contextual variability** | Spaced repetitions are encoded with more varied context, so more retrieval routes overlap with the test context. | Estes (1955); Glenberg (1979); Cepeda et al. (2006): the only theory shown to produce an RI-dependent optimal ISI; Küpper-Tetzel et al. (2014); Cowan et al. (2024) | **Moderate analogue.** A review gradient is computed at a different parameter state and with different neighbouring batches (new facts) at each review. More dispersed reviews sample more diverse parameter states. | Uniform samples a wider range of mid-training states than expanding, which predicts uniform ≥ expanding at a long RI. The effect is likely small with constant LR and LoRA. |
| **Study-phase retrieval / reminding** | A repetition benefits to the extent it retrieves the earlier episode, and more so when that retrieval is effortful but successful. Too long a gap means retrieval fails and the benefit is lost (non-monotonic). | Thios & D'Agostino (1976); Greene (1989, two-process account); Benjamin & Tullis (2010); Wahlheim, Maddox & Jacoby (2014: reminding "sufficient but not necessary"); Toppino et al. (2018); Gerbier & Koenig (2012) | **Partial analogue.** The item is partially forgotten (loss has risen) but its representation still overlaps and its gradient aligns with the original learning direction. There is no explicit retrieval *act*. One could probe "reminding" via cosine similarity between review and Stage-1 gradients. | Predicts benefits for reviews when items are partly but not wholly forgotten. In P4, old-fact loss *decreased* during Stage 2, so the "partly forgotten" state rarely arises before the buffer. |
| **Consolidation** | The second presentation inherits or benefits from consolidation of the first trace, and sleep/time-dependent stabilisation matters. | Cepeda et al. (2006) p. 369 (Wickelgren, 1972); Wixted (2004); Smolen, Zhang & Byrne (2016, cellular mechanisms) | **Weak analogue.** There is no time-dependent stabilisation in SGD with constant LR (deliberately chosen in P4). Loose analogues are LR decay, weight averaging/EMA and replay. | No prediction under constant LR, which is consistent with the null. |
| **Desirable difficulties / retrieval effort; New Theory of Disuse** | Conditions that lower immediate retrieval strength can increase storage strength. Effortful successful retrieval yields bigger gains. | Bjork (1994); Bjork & Bjork (1992, 2011/2014); Pyc & Rawson (2009) | **Partial analogue.** "Difficulty" maps to the loss at review time and "successful retrieval" has no analogue. Under teacher forcing the model is always given the answer, like restudy with feedback. | The expanding schedule's rationale (keep early retrieval successful) is moot. Uniform's later, "harder" reviews give larger gradients but not "retrieval". This predicts a small or null difference. |
| **Multiscale / ACT-R decay models** | Each exposure adds a decaying trace, and decay depends on activation at the time of exposure (ACT-R) or on multiple time-scales (MCM). | Pavlik & Anderson (2005); Mozer et al. (2009); Raaijmakers (2003, SAM) | **Analogue as a measurement model.** These models can be fit to LM loss trajectories to predict optimal review timing. | ACT-R predicts benefits from reviewing at lower activation, which favours later and wider gaps. MCM predicts that schedule effects grow with RI (Latimier et al., 2021, p. 979, citing Lindsey et al., 2009). |

**Entries for theory sources** (the definitional ones are in §1):
- **Hintzman (1974).** Hintzman, D. L. (1974). Theoretical implications of the spacing effect. In R. L. Solso (Ed.), *Theories in cognitive psychology: The Loyola Symposium* (pp. 77–99). Erlbaum. Reissued by Routledge in 2024: https://doi.org/10.4324/9781032722375-5. [V-meta]
  - **Use.** Classic statement of deficient processing / inattention.
- **Glenberg (1979).** Glenberg, A. M. (1979). Component-levels theory of the effects of spacing of repetitions on recall and recognition. *Memory & Cognition, 7*(2), 95–112. https://doi.org/10.3758/BF03197590. [V-meta]
  - **Use.** Encoding (contextual) variability theory.
- **Estes (1955).** Estes, W. K. (1955). Statistical theory of spontaneous recovery and regression. *Psychological Review, 62*(3), 145–154. https://doi.org/10.1037/h0048509. [V-meta]
  - **Use.** Stimulus-fluctuation basis of contextual variability.
- **Thios & D'Agostino (1976).** Thios, S. J., & D'Agostino, P. R. (1976). Effects of repetition as a function of study-phase retrieval. *Journal of Verbal Learning and Verbal Behavior, 15*(5), 529–536. https://doi.org/10.1016/0022-5371(76)90047-5. [V-meta]
  - **Use.** Cepeda et al. (2006, p. 370) note that a lag effect appears only when retrieval of the first presentation is required.
- **Greene (1989).** Greene, R. L. (1989). Spacing effects in memory: Evidence for a two-process account. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 15*(3), 371–377. https://doi.org/10.1037/0278-7393.15.3.371. [V-meta]
  - **Use.** Hybrid account: deficient processing for recognition, study-phase retrieval and contextual variability for recall.
- **Benjamin & Tullis (2010).** Benjamin, A. S., & Tullis, J. G. (2010). What makes distributed practice effective? *Cognitive Psychology, 61*(3), 228–247. https://doi.org/10.1016/j.cogpsych.2010.05.004. [V-meta]
  - **Use.** A critique of encoding-variability-only accounts and an argument for reminding.
- **Wahlheim, Maddox & Jacoby (2014).** Wahlheim, C. N., Maddox, G. B., & Jacoby, L. L. (2014). The role of reminding in the effects of spaced repetitions on cued recall: Sufficient but not necessary. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 40*(1), 94–105. https://doi.org/10.1037/a0034055. [V-abs]
- **Maddox (2016).** Maddox, G. B. (2016). Understanding the underlying mechanism of the spacing effect in verbal learning: A case for encoding variability and study-phase retrieval. *Journal of Cognitive Psychology, 28*(6), 684–706. https://doi.org/10.1080/20445911.2016.1181637. [V-meta]
  - **Use.** Recent review favouring a hybrid account.
- **Pyc & Rawson (2009).** Pyc, M. A., & Rawson, K. A. (2009). Testing the retrieval effort hypothesis: Does greater difficulty correctly recalling information lead to higher levels of memory? *Journal of Memory and Language, 60*(4), 437–447. https://doi.org/10.1016/j.jml.2009.01.004. [V-meta]
- **Raaijmakers (2003).** Raaijmakers, J. G. W. (2003). Spacing and repetition effects in human memory: Application of the SAM model. *Cognitive Science, 27*(3), 431–452. https://doi.org/10.1207/s15516709cog2703_5. [V-meta]
- **Smolen, Zhang & Byrne (2016).** Smolen, P., Zhang, Y., & Byrne, J. H. (2016). The right time to learn: Mechanisms and optimization of spaced learning. *Nature Reviews Neuroscience, 17*(2), 77–88. https://doi.org/10.1038/nrn.2015.18. [V-meta]
  - **Use.** Consolidation-level mechanisms. Notes that models predict benefits from *irregular* intertrial intervals.
- **Bjork (1994).** Bjork, R. A. (1994). Memory and metamemory considerations in the training of human beings. In J. Metcalfe & A. P. Shimamura (Eds.), *Metacognition: Knowing about knowing* (pp. 185–205). MIT Press. https://www.researchgate.net/publication/305433736_Memory_and_Meta-memory_Considerations_in_the_Training_of_Human_Beings. [V-meta]
- **Bjork & Bjork (2011).** Bjork, E. L., & Bjork, R. A. (2011). Making things hard on yourself, but in a good way: Creating desirable difficulties to enhance learning. In M. A. Gernsbacher, R. W. Pew, L. M. Hough, & J. R. Pomerantz (Eds.), *Psychology and the real world: Essays illustrating fundamental contributions to society* (pp. 56–64). Worth. https://www.researchgate.net/publication/284097727. [V-meta. Quotes in this file were checked against the 2nd ed. (2014, pp. 59–68): https://burrell.edu/wp-content/uploads/2020/09/EBjorkRBjork_FABBSchapter2014-2nd-ed._WithCoverPage.pdf. The full 2011 editor list is **UNVERIFIED**; Gernsbacher and Pomerantz are confirmed.]
- **Bjork & Bjork (1992), New Theory of Disuse.** In A. Healy, S. Kosslyn, & R. Shiffrin (Eds.), *From learning processes to cognitive processes: Essays in honor of William K. Estes* (Vol. 2, pp. 35–67). Erlbaum. **UNVERIFIED** beyond the reference list of Bjork & Bjork (2014). No URL found.

**Takeaway for the paper.** The intro says the human mechanisms "do not have clean analogues" in SGD. That is too strong in one direction and too weak in another.
- *Deficient processing* and the *storage/retrieval-strength* dynamic do have a fairly clean analogue: gradient size scales with current loss.
- *Study-phase retrieval*, the account most often invoked for expanding schedules (Gerbier, Toppino), and the *retrieval-success* rationale for expanding retrieval (Landauer & Bjork; Karpicke & Roediger) have no analogue under teacher-forced restudy.
- Cite Cepeda et al. (2006, pp. 368–370) for the four theories. Cite Bjork (1994) and Pyc & Rawson (2009) for "retrieval difficulty"; Cepeda 2006 does not list it.

---

## 5. Degree of original learning vs. rate of forgetting (P4 §3.3)

**Slamecka & McElree (1983).** Slamecka, N. J., & McElree, B. (1983). Normal forgetting of verbal lists as a function of their degree of learning. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 9*(3), 384–397. https://doi.org/10.1037/0278-7393.9.3.384. [V-meta; content via Rivera-Lares et al. 2022 full text and Semantic Scholar]
- **Finding.** Three experiments: categorised lists, paired associates, and sentences (verbatim and gist). Tests came soon after learning, at 1 day and at 5 days. More study trials raised the *intercept* but not the *slope* of the forgetting curves: "forgetting … is independent of their degree of learning."
- **Use.** The direct human parallel to "review changed the starting point, not the forgetting rate." Cite it, along with the caveats below.

**Loftus (1985).** Loftus, G. R. (1985). Evaluating forgetting curves. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 11*(2), 397–406. https://doi.org/10.1037/0278-7393.11.2.397. [V-meta; content via Rivera-Lares et al. 2022 and the abstract as quoted in search]
- **Finding.** Vertical comparison (equal slopes at the same delays) is scale-dependent. Any monotone transform of the dependent measure can create or remove an interaction. Loftus proposes the *horizontal* comparison: the time needed to fall from one performance level to another. Applying it to a variety of data, he concluded that forgetting is *slower* for higher degrees of learning.
- **Use.** A key caveat for P4.
  - P4's metric is cross-entropy, −log p(answer). Parallel curves in loss mean *constant ratios* of answer probability. On the probability or accuracy scale the same data would not be parallel.
  - A Loftus-style horizontal reading of P4's Table 2 is informative. The review arms' loss *after* 180 buffer steps (3.809) is still below the no-review arm's loss at buffer *start* (3.830). In horizontal terms, review is worth more than 180 steps of buffer retention, which is a more striking way to state the result.

**Slamecka (1985).** Slamecka, N. J. (1985). On comparing rates of forgetting: Comment on Loftus (1985). *Journal of Experimental Psychology: Learning, Memory, and Cognition, 11*(4), 812–816. https://doi.org/10.1037/0278-7393.11.1-4.812. [V-meta]
- **Finding.** The horizontal method confounds memory *age*: one group is measured at a later point on its curve.
- **Use.** The age confound applies to P4 in reverse. At buffer start, review-arm facts were last trained about 15 steps earlier, while no-review facts were last trained about 195+ steps earlier, at the end of Stage 1. Jost's second law, discussed in Rivera-Lares et al. (2022) and Wixted (2004), says that of two equally strong memories the older decays more slowly. Equal slopes for "younger" review-arm memories are therefore not straightforwardly "same durability."

**Bogartz (1990).** Bogartz, R. S. (1990). Evaluating forgetting curves psychologically. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 16*(1), 138–148. https://doi.org/10.1037/0278-7393.16.1.138. [V-meta; content via Loftus & Bamber 1990 description]
- **Finding.** Proposes a definition of learning–forgetting independence under which independence holds in the existing data, and criticises the horizontal method.
- Reply: Loftus, G. R., & Bamber, D. (1990). Learning–forgetting independence, unidimensional memory models, and feature models: Comment on Bogartz (1990). *JEP:LMC, 16*(5), 916–926. https://doi.org/10.1037/0278-7393.16.5.916. [V-meta]
- **Use.** Whether forgetting rate "depends on" learning is partly definitional. P4 should state its operational definition (loss difference over a fixed number of steps on the CE scale) and avoid mechanistic language ("does not install a more durable memory").

**Wixted (2004).** Wixted, J. T. (2004). The psychology and neuroscience of forgetting. *Annual Review of Psychology, 55*, 235–269. https://doi.org/10.1146/annurev.psych.55.090902.141555. [V-meta + abstract via search]
- **Finding.** Forgetting reflects interference with not-yet-consolidated traces from *any* subsequent mental activity, not only similar material. Forgetting curves show an ever-decreasing proportional rate (Jost's law), which Rivera-Lares et al. cite in support of vertical comparison.
- See also Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting. *Psychological Science, 2*(6), 409–415. https://doi.org/10.1111/j.1467-9280.1991.tb00175.x [V-meta], on power-law forgetting.
- **Use.** Frames P4's buffer as retroactive interference from new-fact training. Power-law forgetting implies that "forgetting over 180 steps" depends on where on the curve one starts.

**Rivera-Lares et al. (2022).** Rivera-Lares, K., Logie, R., Baddeley, A., & Della Sala, S. (2022). Rate of forgetting is independent of initial degree of learning. *Memory & Cognition, 50*(8), 1706–1718. https://doi.org/10.3758/s13421-021-01271-1. [V-full via Europe PMC]
- **Finding.** Four experiments: 36 sentences presented 2, 4 or 6 times; cued recall at 30 s, 1 day and 1 week (or 3 days). Initial performance increased with repetitions, but "the rate of forgetting proved to be independent of initial acquisition."
  - They analysed the data with Bayesian generalised linear mixed models (brms), i.e., no interaction on the model's link scale.
  - They explicitly adopt Slamecka & McElree's vertical comparison to avoid comparing memories of different ages.
- **Use.** A modern replication of "parallel curves". It is still a choice of scale (logit), which supports reporting P4 on more than one scale.

**Rivera-Lares et al. (2023).** Rivera-Lares, K., Della Sala, S., Baddeley, A., & Logie, R. (2023). Rate of forgetting is independent from initial degree of learning across different age groups. *Quarterly Journal of Experimental Psychology, 76*(7), 1672–1682. https://doi.org/10.1177/17470218221128780. [V-meta]

**Rivera-Lares et al. (2025).** Rivera-Lares, K., Baddeley, A., & Della Sala, S. (2025). Influence of degree of learning on rate of forgetting of tonal sequences. *Memory & Cognition, 53*(2), 682–691. https://doi.org/10.3758/s13421-024-01597-6. [V-meta + PubMed summary via search]
- **Finding.** Same result for nonverbal tonal sequences presented once vs three times.

**Contrasting human evidence: schedule and testing *do* change forgetting slopes**
- K&R (2007), Exp 1: expanding fell from .71 to .33 between the 10-min and 2-day tests, while equal fell from .62 to .45. This is a crossover, i.e., faster forgetting after expanding.
- Kang et al. (2014): *less* forgetting after expanding (.13 vs .19).
- Roediger & Karpicke (2006): restudy is better at 5 min and testing is better at 1 week.
- Walsh et al. (2018): spacing "increased retention at the start of tests, and accelerated relearning."

So in humans, *more repetitions of the same kind* shift the intercept (Slamecka & McElree; Rivera-Lares), while *different kinds* of practice (spacing, testing) can change slope. P4's finding that review shifts the intercept but not the slope is what the human literature predicts if gradient review acts like additional restudy repetitions rather than a qualitatively different encoding.

**Statistical caveat specific to P4.** The buffer-forgetting CIs for review vs. no review are wide: uniform − none = +0.0097 [−0.072, +0.091], against about 0.15 total buffer forgetting. P4 therefore cannot rule out review changing the forgetting rate by roughly ±50%. "Indistinguishable from zero" is accurate, but "review does not install a more durable memory" is not supported. Only the expanding − uniform forgetting difference is tightly bounded ([−0.012, +0.006]).

---

## 6. Testing effect vs. restudy: the key disanalogy

**Roediger & Karpicke (2006).** Roediger, H. L., III, & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science, 17*(3), 249–255. https://doi.org/10.1111/j.1467-9280.2006.01693.x. [V-full]
- **Finding.**
  - Exp 1: at 5 min, restudy beat testing (81% vs 75%, d = 0.52). At 2 days, testing won (68% vs 54%, d = 0.95), and at 1 week (56% vs 42%, d = 0.83).
  - Exp 2: at 5 min, SSSS 83% > SSST 78% > STTT 71%. At 1 week, STTT 61% > SSST 56% > SSSS 40%.
  - Restudy increased confidence.
- **Use.** Gradient training on the target string is restudy (SSSS-like). Human evidence says restudy has the *least* durable benefit. P4's review is therefore an instance of the weakest human review mode.

**Rowland (2014).** Rowland, C. A. (2014). The effect of testing versus restudy on retention: A meta-analytic review of the testing effect. *Psychological Bulletin, 140*(6), 1432–1463. https://doi.org/10.1037/a0037559. [V-abs; numbers via the full text as quoted in search and in Latimier p. 959]
- **Finding.** g = 0.50 [0.42, 0.58] from 159 effect sizes in 61 studies. With feedback g ≈ 0.73, without ≈ 0.39. RI ≥ 24 h g = 0.69 vs < 24 h g = 0.41. The evidence supports effortful processing; recall tests beat recognition tests.
- **Use.** Quantifies what P4's restudy-only review omits.

**Adesope, Trevisan & Sundararajan (2017).** Adesope, O. O., Trevisan, D. A., & Sundararajan, N. (2017). Rethinking the use of tests: A meta-analysis of practice testing. *Review of Educational Research, 87*(3), 659–701. https://doi.org/10.3102/0034654316689306. [V-meta]
- **Finding (via Latimier p. 959).** g = 0.61 [0.58, 0.65].

**Karpicke & Roediger (2008).** Karpicke, J. D., & Roediger, H. L., III. (2008). The critical importance of retrieval for learning. *Science, 319*(5865), 966–968. https://doi.org/10.1126/science.1152408. [V-meta; content not re-verified]
- **Use.** Commonly cited for the claim that repeated retrieval, not repeated study, drives retention.

**How to use §6 in the paper.** Add an explicit "disanalogy" paragraph:
1. In LM continued training, "review" is teacher-forced restudy. The model receives the full target every time, so there is never retrieval failure and never feedback-free retrieval.
2. The best-supported human rationale for expanding schedules is maintaining retrieval *success* while retrieval difficulty grows. It applies to test-type practice and was explicitly predicted *not* to apply to repetition-type practice (Landauer & Bjork, 1978, p. 626). Landauer & Bjork found uniform ≥ expanding for repetitions.
3. The P4 null is therefore what the human literature predicts for the restudy case.
4. A genuinely retrieval-like LM review would have the model generate an answer and then be updated by correctness (e.g., an RL or self-distillation signal). That is a natural follow-up.

---

## 7. Computational models of spacing and optimal scheduling

- **Pavlik & Anderson (2005).** Pavlik, P. I., Jr., & Anderson, J. R. (2005). Practice and forgetting effects on vocabulary memory: An activation-based model of the spacing effect. *Cognitive Science, 29*(4), 559–586. https://doi.org/10.1207/s15516709cog0000_14. [V-meta + abstract via search]
  - **Finding.** In ACT-R, each practice adds a power-decaying trace whose decay rate depends on activation at the time of practice. The relative benefit of spacing grew with more practice and longer RIs.
  - **Use.** Fit to LM loss trajectories as a predictive model.
- **Pavlik & Anderson (2008).** Pavlik, P. I., Jr., & Anderson, J. R. (2008). Using a model to compute the optimal schedule of practice. *Journal of Experimental Psychology: Applied, 14*(2), 101–117. https://doi.org/10.1037/1076-898X.14.2.101. [V-abs]
  - **Finding.** An item-level scheduler maximised long-term gain per unit of practice time and produced large effect sizes. "As practice repetitions accumulate for each item … this optimal interval increases."
  - **Use.** Gives model-based expanding schedules a principled, per-item, adaptive basis. P4's fixed population-level expansion ratio of 1.35 is not that.
- **Mozer et al. (2009), MCM.** Mozer, M. C., Pashler, H., Cepeda, N., Lindsey, R., & Vul, E. (2009). Predicting the optimal spacing of study: A multiscale context model of memory. In Y. Bengio et al. (Eds.), *Advances in Neural Information Processing Systems 22* (pp. 1321–1329). https://proceedings.neurips.cc/paper_files/paper/2009/hash/6bc24fc1ab650b25b4114e93a98f1eba-Abstract.html. [V-full]
  - **Finding.** A cascade of leaky integrators across time-scales, synthesising Staddon et al. and Raaijmakers' SAM. It predicts the non-monotonic ISI × RI pattern from single-session forgetting data, and appropriate spacing "can double retention."
  - **Use.** A NeurIPS-native bridge to cite. Per Latimier, MCM predicts expanding advantages that grow with RI, which was not supported meta-analytically.
- **Lindsey et al. (2014).** Lindsey, R. V., Shroyer, J. D., Pashler, H., & Mozer, M. C. (2014). Improving students' long-term knowledge retention through personalized review. *Psychological Science, 25*(3), 639–647. https://doi.org/10.1177/0956797613504302. [V-abs]
  - **Finding.** In a semester-long classroom study, personalised review gave a 16.5% boost over massed study and 10.0% over one-size-fits-all spaced review.
  - **Use.** Gains come from personalisation (item × learner state), not schedule shape.
- **Walsh et al. (2018), PPE.** Walsh, M. M., Gluck, K. A., Gunzelmann, G., Jastrzembski, T., Krusmark, M., Myung, J. I., Pitt, M. A., & Zhou, R. (2018). Mechanisms underlying the spacing effect in learning: A comparison of three computational models. *Journal of Experimental Psychology: General, 147*(9), 1325–1348. https://doi.org/10.1037/xge0000416. [V-abs]
  - **Finding.** Spacing "slowed learning during the acquisition phase, increased retention at the start of tests, and accelerated relearning". Only the Predictive Performance Equation captured spacing-accelerated relearning.
  - **Use.** Suggests a *relearning/savings* probe after the buffer as an additional LM metric.
- **Settles & Meeder (2016), half-life regression.** Settles, B., & Meeder, B. (2016). A trainable spaced repetition model for language learning. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1848–1858). https://doi.org/10.18653/v1/P16-1174. [V-abs]
  - **Finding.** HLR estimates per-item memory half-life from Duolingo logs, "reducing error by 45%+" vs baselines, and improved daily engagement by 12%.
  - **Use.** A per-fact half-life could be fit to LM loss as a compact descriptive measure.
- **Reddy et al. (2016).** Reddy, S., Labutov, I., Banerjee, S., & Joachims, T. (2016). Unbounded human learning: Optimal scheduling for spaced repetition. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1815–1824). https://doi.org/10.1145/2939672.2939850. [V-abs via arXiv:1602.07032]
  - **Finding.** A queueing-network model of the Leitner system, with a predicted and confirmed phase transition in outcomes as the rate of new-item introduction increases.
  - **Use.** Directly analogous to the old/new data-mixing trade-off in continued training (P4 §3.4, the plasticity cost).
- **Tabibian et al. (2019).** Tabibian, B., Upadhyay, U., De, A., Zarezade, A., Schölkopf, B., & Gomez-Rodriguez, M. (2019). Enhancing human learning via spaced repetition optimization. *PNAS, 116*(10), 3988–3993. https://doi.org/10.1073/pnas.1815156116. [V-abs]
  - **Finding.** Spaced repetition is cast as optimal control of marked temporal point processes. For two memory models, the optimal review intensity equals the current recall probability (MEMORIZE), and it beat heuristics in Duolingo data.
  - **Use.** Suggests a loss-proportional replay rate as the principled LM analogue.
- **Ye, Su & Cao (2022), SSP-MMC.** Ye, J., Su, J., & Cao, Y. (2022). A stochastic shortest path algorithm for optimizing spaced repetition scheduling. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 4381–4390). https://doi.org/10.1145/3534678.3539081. [V-abs]
  - **Finding.** 220M review logs (MaiMemo), a Markov memory model, and a stochastic-shortest-path scheduler minimising review cost, with a 12.6% improvement over the prior state of the art.
  - **Extension.** Su, J., Ye, J., Nie, L., Cao, Y., & Chen, Y. (2023). Optimizing spaced repetition schedule by capturing the dynamics of memory. *IEEE TKDE, 35*(10), 10085–10097. https://doi.org/10.1109/TKDE.2023.3251721. [V-meta]
  - **Note.** "FSRS" (used in Anki) is the open-source descendant. FSRS itself has no peer-reviewed paper that I verified (**UNVERIFIED** as a citable source).
  - **Use.** If the paper mentions Anki, cite SM-2 (Wozniak) and this line of work.
- **Eglington & Pavlik (2020).** Eglington, L. G., & Pavlik, P. I., Jr. (2020). Optimizing practice scheduling requires quantitative tracking of individual item performance. *npj Science of Learning, 5*, 15. https://doi.org/10.1038/s41539-020-00074-4. [V-abs]
  - **Finding.** Fixed spacing recommendations are "inherently suboptimal", and model-based item tracking gave up to 40% more items recalled.
- **ML bridge (outside this file's scope; verify in the ML review).** Amiri, H., Miller, T., & Savova, G. (2017). Repeat before forgetting: Spaced repetition for efficient and effective training of neural networks. In *Proceedings of EMNLP 2017* (pp. 2401–2410). https://doi.org/10.18653/v1/D17-1255. [V-meta]

---

## 8. Synthesis

### 8.1 What the human literature actually predicts for expanding vs. uniform at long delays

1. **On average, nothing.** For retrieval practice, expanding ≈ uniform: g = 0.034 [−0.10, 0.17], I² = 0% (Latimier et al., 2021). Cepeda et al. (2006) also found no significant difference (62.0% vs 58.6%).
   - Retention interval does not moderate this (Latimier β = 0.03).
   - Absolute spacing matters a great deal (Karpicke & Bauernschmidt, 2011: about 200%; Cepeda et al., 2008, 2009) but relative spacing does not.
2. **Where individual studies diverge, the moderators are mechanistic:**
   - (a) The *first-interval confound*. Equal schedules usually delay the first test, and once that is equated, schedule shape stops mattering (Karpicke & Roediger, 2007, Exp 3).
   - (b) *Fast forgetting or high interference* favours expanding, because early tests stay successful (Storm et al., 2010; Maddox et al., 2011).
   - (c) *Low initial learning* favours expanding for restudy across days (Toppino et al., 2018; Gerbier et al., 2015).
   - (d) *More exposures* trend toward expanding (Latimier moderator, p = .09).
   - (e) *Outcome measure.* Expanding yields higher average accessibility *during* training even when final retention is equal (Kang et al., 2014; Balota et al., 2006), and it can be more efficient (Yan et al., 2020).
3. **For restudy (repetition-type) practice, which is what LM gradient review is**, Landauer & Bjork (1978) predicted and found uniform ≥ expanding at a short delay. The multi-day restudy literature is mixed:
   - expanding ≥ uniform in Gerbier & Koenig (2012) and Gerbier et al. (2015);
   - equal ≈ expanding at a 35-day RI in Küpper-Tetzel et al. (2014);
   - expanding better only after weak initial training in Toppino et al. (2018).

### 8.2 How P4's result relates

- P4 matched the first review (step 8), the last review (step 165), the count (12) and the compute. It varied only relative spacing and tested one long delay.
- Under exactly these conditions, the human literature (K&R 2007 Exp 3; K&B 2011; Latimier 2021; Mettler et al. 2016) predicts **no difference**. P4 found none, with a tight CI ([−0.0053, +0.0041] vs a review effect of −0.17).
- The result is *consistent with* the human literature. It does not show a new transfer of a human phenomenon.
- P4's trajectory advantage for expanding during Stage 2 is the analogue of Kang et al.'s (2014) average-recallability advantage. It is a known human pattern, not merely an artifact.
- P4's "review shifts the intercept, not the slope" result parallels Slamecka & McElree (1983) and Rivera-Lares et al. (2022). It carries their scale-dependence caveats (Loftus, 1985), and the memory-age confound runs in the opposite direction here.
- Two P4-specific facts weaken the mapping to human schedules:
  - Per individual fact there is about 0.5 review exposure on average, with most facts reviewed 0–1 times (§0). P4 manipulates the timing of *population-level replay batches*, not per-item expanding schedules.
  - Old-fact loss fell during Stage 2 even without review, so "retention" is entangled with shared format learning.

### 8.3 Claims in the current intro, abstract and conclusion that overreach (with suggested fixes)

1. **Intro: "expanding-interval schedules … are shown to greatly improve memory in humans (Landauer & Bjork, 1978; Cepeda et al., 2006)."**
   - *Overreach and miscitation.* L&B found about a 10-point advantage at a 30-min delay, for test-type practice only, and found uniform slightly better for repetitions. Cepeda (2006) found expanding vs fixed non-significant and "necessarily tentative."
   - *Fix:* "Expanding schedules were proposed as optimal for test-type rehearsal (Landauer & Bjork, 1978), but quantitative syntheses find no reliable advantage over equal spacing (Cepeda et al., 2006; Latimier et al., 2021)."
2. **Intro: mechanisms "consolidation, retrieval difficulty, encoding variability (Cepeda et al., 2006) … do not have clean analogues."**
   - Cepeda lists deficient processing, encoding variability, consolidation and study-phase retrieval. "Retrieval difficulty" comes from Bjork (1994) and Pyc & Rawson (2009).
   - Deficient processing has a fairly clean analogue: gradient magnitude scales with loss. Study-phase retrieval and retrieval success do not.
   - *Fix:* name the theories correctly and state which ones transfer.
3. **Framing: the paper invokes the spacing effect, but the design has no massed-review control.** All review arms are spaced, so "review helps" is a repetition/replay effect, not a spacing effect.
   - *Fix:* add a massed arm (12 consecutive review steps) and/or vary absolute spacing (K&B 2011), or reword to "relative spacing (schedule shape)."
4. **Conclusion: "equally spaced practice matches or beats expanding retrieval once memory is tested after a long delay (K&R 2007); the contribution here is showing that this human phenomenon reproduces in LLM continued training."**
   - (a) The K&R "beats" result is driven by first-test delay, which P4 deliberately equated, so P4 corresponds to K&R Exp 3 (no difference).
   - (b) The meta-analytic consensus is "no difference, no RI moderation," not "equal beats expanding at long delays."
   - (c) There are long-delay counterexamples favouring expanding, mostly with restudy over days (Gerbier et al., 2015; Toppino et al., 2018 at low initial training; Kang et al., 2014 on forgetting and average recall).
   - (d) LM review is restudy, not retrieval practice.
   - (e) P4 does not implement per-item schedules.
   - (f) A null across 3 seeds is consistency, not reproduction. The CI does support practical equivalence; state the equivalence margin (TOST).
   - *Fix:* "Our null is what the human literature predicts when first- and last-review timing are matched and only relative spacing varies (Karpicke & Roediger, 2007, Exp. 3; Karpicke & Bauernschmidt, 2011; Latimier et al., 2021)."
5. **Conclusion: "how you space it does not [matter]" and "uniform review is the sensible default."**
   - Only one expansion ratio (1.35), one RI, one model and population-level replay were tested.
   - Human data show that absolute spacing and the gap/RI ratio matter (Cepeda et al., 2008). Adaptive schedules beat both fixed shapes (Mettler et al., 2016; Lindsey et al., 2014; Pavlik & Anderson, 2008).
   - If the model must perform *during* continued training, expanding's higher average retention (Kang et al., 2014; P4's own trajectory metric; the P4 one-pager's AUC result in Pythia-162M) is a real benefit.
   - *Fix:* "Under matched endpoints and a single long delay, schedule shape did not affect final retention."
6. **§3.3: "review does not appear to install a more durable memory that decays more slowly; it simply starts the buffer from a lower loss."**
   - The review-vs-none forgetting CIs (±0.08) are about half the size of the total forgetting (0.15), so slope equality is not established.
   - Parallelism on the CE (log-probability) scale is scale-specific (Loftus, 1985).
   - Review-arm memories are younger at buffer start (the age confound; Slamecka, 1985).
   - *Fix:* report "buffer forgetting on the loss scale was not detectably different (CI …)", cite Slamecka & McElree (1983) and Rivera-Lares et al. (2022) as the human parallel, and add the Loftus caveat. Optionally add the horizontal statement that review arms after 180 buffer steps are still better than no-review at buffer start.
7. **§3.5: expanding's trajectory advantage is "most plausibly an artifact."**
   - It is front-loading, but the human literature treats average accessibility during training as a substantive outcome (Kang et al., 2014; Balota et al., 2006).
   - *Fix:* call it "a during-training benefit that does not persist after a shared final review."

### 8.4 Concrete additions the human literature suggests

All are cheap with the existing code.
- A massed-review arm and an absolute-spacing manipulation.
- Multiple buffer lengths, to test RI × schedule.
- Reporting on probability and accuracy scales alongside loss.
- A per-fact analysis: reviewed vs. never-reviewed facts, and number of reviews.
- A savings/relearning probe after the buffer (Ebbinghaus; Walsh et al., 2018).
- The existing `adaptive_due` condition.
- A retrieval-like review variant: generate, then update by correctness.

---

## 9. Consolidated reference list (APA 7)

- Adesope, O. O., Trevisan, D. A., & Sundararajan, N. (2017). Rethinking the use of tests: A meta-analysis of practice testing. *Review of Educational Research, 87*(3), 659–701. https://doi.org/10.3102/0034654316689306
- Amiri, H., Miller, T., & Savova, G. (2017). Repeat before forgetting: Spaced repetition for efficient and effective training of neural networks. In *Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing* (pp. 2401–2410). https://doi.org/10.18653/v1/D17-1255
- Balota, D. A., Duchek, J. M., & Logan, J. M. (2007). Is expanded retrieval practice a superior form of spaced retrieval? A critical review of the extant literature. In J. S. Nairne (Ed.), *The foundations of remembering: Essays in honor of Henry L. Roediger III* (pp. 83–105). Psychology Press. http://psychnet.wustl.edu/coglab/wp-content/uploads/2015/01/2007-Is-expanded.pdf
- Balota, D. A., Duchek, J. M., Sergent-Marshall, S. D., & Roediger, H. L., III. (2006). Does expanded retrieval produce benefits over equal-interval spacing? Explorations of spacing effects in healthy aging and early stage Alzheimer's disease. *Psychology and Aging, 21*(1), 19–31. https://doi.org/10.1037/0882-7974.21.1.19
- Benjamin, A. S., & Tullis, J. G. (2010). What makes distributed practice effective? *Cognitive Psychology, 61*(3), 228–247. https://doi.org/10.1016/j.cogpsych.2010.05.004
- Bjork, E. L., & Bjork, R. A. (2011). Making things hard on yourself, but in a good way: Creating desirable difficulties to enhance learning. In M. A. Gernsbacher et al. (Eds.), *Psychology and the real world* (pp. 56–64). Worth. https://www.researchgate.net/publication/284097727 (2nd ed., 2014, pp. 59–68: https://burrell.edu/wp-content/uploads/2020/09/EBjorkRBjork_FABBSchapter2014-2nd-ed._WithCoverPage.pdf)
- Bjork, R. A. (1994). Memory and metamemory considerations in the training of human beings. In J. Metcalfe & A. P. Shimamura (Eds.), *Metacognition: Knowing about knowing* (pp. 185–205). MIT Press. https://www.researchgate.net/publication/305433736
- Bogartz, R. S. (1990). Evaluating forgetting curves psychologically. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 16*(1), 138–148. https://doi.org/10.1037/0278-7393.16.1.138
- Carpenter, S. K., Cepeda, N. J., Rohrer, D., Kang, S. H. K., & Pashler, H. (2012). Using spacing to enhance diverse forms of learning: Review of recent research and implications for instruction. *Educational Psychology Review, 24*(3), 369–378. https://doi.org/10.1007/s10648-012-9205-z
- Cepeda, N. J., Coburn, N., Rohrer, D., Wixted, J. T., Mozer, M. C., & Pashler, H. (2009). Optimizing distributed practice: Theoretical analysis and practical implications. *Experimental Psychology, 56*(4), 236–246. https://doi.org/10.1027/1618-3169.56.4.236
- Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin, 132*(3), 354–380. https://doi.org/10.1037/0033-2909.132.3.354
- Cepeda, N. J., Vul, E., Rohrer, D., Wixted, J. T., & Pashler, H. (2008). Spacing effects in learning: A temporal ridgeline of optimal retention. *Psychological Science, 19*(11), 1095–1102. https://doi.org/10.1111/j.1467-9280.2008.02209.x
- Cowan, E. T., Zhang, Y., Rottman, B. M., & Murty, V. P. (2024). The effects of mnemonic variability and spacing on memory over multiple timescales. *Proceedings of the National Academy of Sciences, 121*(12), e2311077121. https://doi.org/10.1073/pnas.2311077121
- Cull, W. L. (2000). Untangling the benefits of multiple study opportunities and repeated testing for cued recall. *Applied Cognitive Psychology, 14*(3), 215–235. https://doi.org/10.1002/(SICI)1099-0720(200005/06)14:3%3C215::AID-ACP640%3E3.0.CO;2-1
- Donovan, J. J., & Radosevich, D. J. (1999). A meta-analytic review of the distribution of practice effect: Now you see it, now you don't. *Journal of Applied Psychology, 84*(5), 795–805. https://doi.org/10.1037/0021-9010.84.5.795
- Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham, D. T. (2013). Improving students' learning with effective learning techniques: Promising directions from cognitive and educational psychology. *Psychological Science in the Public Interest, 14*(1), 4–58. https://doi.org/10.1177/1529100612453266
- Ebbinghaus, H. (1913). *Memory: A contribution to experimental psychology* (H. A. Ruger & C. E. Bussenius, Trans.). Teachers College, Columbia University. (Original work published 1885) https://psychclassics.yorku.ca/Ebbinghaus/index.htm
- Eglington, L. G., & Pavlik, P. I., Jr. (2020). Optimizing practice scheduling requires quantitative tracking of individual item performance. *npj Science of Learning, 5*, 15. https://doi.org/10.1038/s41539-020-00074-4
- Estes, W. K. (1955). Statistical theory of spontaneous recovery and regression. *Psychological Review, 62*(3), 145–154. https://doi.org/10.1037/h0048509
- Gerbier, E., & Koenig, O. (2012). Influence of multiple-day temporal distribution of repetitions on memory: A comparison of uniform, expanding, and contracting schedules. *Quarterly Journal of Experimental Psychology, 65*(3), 514–525. https://doi.org/10.1080/17470218.2011.600806
- Gerbier, E., & Toppino, T. C. (2015). The effect of distributed practice: Neuroscience, cognition, and education. *Trends in Neuroscience and Education, 4*(3), 49–59. https://doi.org/10.1016/j.tine.2015.01.001
- Gerbier, E., Toppino, T. C., & Koenig, O. (2015). Optimising retention through multiple study opportunities over days: The benefit of an expanding schedule of repetitions. *Memory, 23*(6), 943–954. https://doi.org/10.1080/09658211.2014.944916
- Glenberg, A. M. (1979). Component-levels theory of the effects of spacing of repetitions on recall and recognition. *Memory & Cognition, 7*(2), 95–112. https://doi.org/10.3758/BF03197590
- Greene, R. L. (1989). Spacing effects in memory: Evidence for a two-process account. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 15*(3), 371–377. https://doi.org/10.1037/0278-7393.15.3.371
- Hintzman, D. L. (1974). Theoretical implications of the spacing effect. In R. L. Solso (Ed.), *Theories in cognitive psychology: The Loyola Symposium* (pp. 77–99). Erlbaum. (Reissued 2024, Routledge) https://doi.org/10.4324/9781032722375-5
- Kang, S. H. K. (2016). Spaced repetition promotes efficient and effective learning: Policy implications for instruction. *Policy Insights from the Behavioral and Brain Sciences, 3*(1), 12–19. https://doi.org/10.1177/2372732215624708
- Kang, S. H. K., Lindsey, R. V., Mozer, M. C., & Pashler, H. (2014). Retrieval practice over the long term: Should spacing be expanding or equal-interval? *Psychonomic Bulletin & Review, 21*(6), 1544–1550. https://doi.org/10.3758/s13423-014-0636-z
- Karpicke, J. D., & Bauernschmidt, A. (2011). Spaced retrieval: Absolute spacing enhances learning regardless of relative spacing. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 37*(5), 1250–1257. https://doi.org/10.1037/a0023436
- Karpicke, J. D., & Roediger, H. L., III. (2007). Expanding retrieval practice promotes short-term retention, but equally spaced retrieval enhances long-term retention. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 33*(4), 704–719. https://doi.org/10.1037/0278-7393.33.4.704
- Karpicke, J. D., & Roediger, H. L., III. (2008). The critical importance of retrieval for learning. *Science, 319*(5865), 966–968. https://doi.org/10.1126/science.1152408
- Karpicke, J. D., & Roediger, H. L., III. (2010). Is expanding retrieval a superior method for learning text materials? *Memory & Cognition, 38*(1), 116–124. https://doi.org/10.3758/MC.38.1.116
- Kim, A. S. N., Wong-Kee-You, A. M. B., Wiseheart, M., & Rosenbaum, R. S. (2019). The spacing effect stands up to big data. *Behavior Research Methods, 51*(4), 1485–1497. https://doi.org/10.3758/s13428-018-1184-7
- Kim, S. K., & Webb, S. (2022). The effects of spaced practice on second language learning: A meta-analysis. *Language Learning, 72*(1), 269–319. https://doi.org/10.1111/lang.12479
- Küpper-Tetzel, C. E., Kapler, I. V., & Wiseheart, M. (2014). Contracting, equal, and expanding learning schedules: The optimal distribution of learning sessions depends on retention interval. *Memory & Cognition, 42*(5), 729–741. https://doi.org/10.3758/s13421-014-0394-1
- Landauer, T. K., & Bjork, R. A. (1978). Optimum rehearsal patterns and name learning. In M. M. Gruneberg, P. E. Morris, & R. N. Sykes (Eds.), *Practical aspects of memory* (pp. 625–632). Academic Press. https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/Landauer.Bjork_.1978.pdf
- Latimier, A., Peyre, H., & Ramus, F. (2021). A meta-analytic review of the benefit of spacing out retrieval practice episodes on retention. *Educational Psychology Review, 33*(3), 959–987. https://doi.org/10.1007/s10648-020-09572-8
- Leitner, S. (1972). *So lernt man lernen* [How to learn to learn]. Herder. https://openlibrary.org/works/OL9068102W (**1972 date UNVERIFIED** against a primary catalogue)
- Lindsey, R. V., Shroyer, J. D., Pashler, H., & Mozer, M. C. (2014). Improving students' long-term knowledge retention through personalized review. *Psychological Science, 25*(3), 639–647. https://doi.org/10.1177/0956797613504302
- Loftus, G. R. (1985). Evaluating forgetting curves. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 11*(2), 397–406. https://doi.org/10.1037/0278-7393.11.2.397
- Loftus, G. R., & Bamber, D. (1990). Learning–forgetting independence, unidimensional memory models, and feature models: Comment on Bogartz (1990). *Journal of Experimental Psychology: Learning, Memory, and Cognition, 16*(5), 916–926. https://doi.org/10.1037/0278-7393.16.5.916
- Logan, J. M., & Balota, D. A. (2008). Expanded vs. equal interval spaced retrieval practice: Exploring different schedules of spacing and retention interval in younger and older adults. *Aging, Neuropsychology, and Cognition, 15*(3), 257–280. https://doi.org/10.1080/13825580701322171
- Maddox, G. B. (2016). Understanding the underlying mechanism of the spacing effect in verbal learning: A case for encoding variability and study-phase retrieval. *Journal of Cognitive Psychology, 28*(6), 684–706. https://doi.org/10.1080/20445911.2016.1181637
- Maddox, G. B., Balota, D. A., Coane, J. H., & Duchek, J. M. (2011). The role of forgetting rate in producing a benefit of expanded over equal spaced retrieval in young and older adults. *Psychology and Aging, 26*(3), 661–670. https://doi.org/10.1037/a0022942
- Mawson, R. D., & Kang, S. H. K. (2025). The distributed practice effect on classroom learning: A meta-analytic review of applied research. *Behavioral Sciences, 15*(6), 771. https://doi.org/10.3390/bs15060771
- Mettler, E., Massey, C. M., & Kellman, P. J. (2016). A comparison of adaptive and fixed schedules of practice. *Journal of Experimental Psychology: General, 145*(7), 897–917. https://doi.org/10.1037/xge0000170
- Mozer, M. C., Pashler, H., Cepeda, N., Lindsey, R., & Vul, E. (2009). Predicting the optimal spacing of study: A multiscale context model of memory. In Y. Bengio, D. Schuurmans, J. Lafferty, C. K. I. Williams, & A. Culotta (Eds.), *Advances in Neural Information Processing Systems 22* (pp. 1321–1329). https://proceedings.neurips.cc/paper_files/paper/2009/hash/6bc24fc1ab650b25b4114e93a98f1eba-Abstract.html
- Murray, E., Horner, A. J., & Göbel, S. M. (2025). A meta-analytic review of the effectiveness of spacing and retrieval practice for mathematics learning. *Educational Psychology Review, 37*, Article 75. https://doi.org/10.1007/s10648-025-10035-1
- Murre, J. M. J., & Dros, J. (2015). Replication and analysis of Ebbinghaus' forgetting curve. *PLOS ONE, 10*(7), e0120644. https://doi.org/10.1371/journal.pone.0120644
- Pavlik, P. I., Jr., & Anderson, J. R. (2005). Practice and forgetting effects on vocabulary memory: An activation-based model of the spacing effect. *Cognitive Science, 29*(4), 559–586. https://doi.org/10.1207/s15516709cog0000_14
- Pavlik, P. I., Jr., & Anderson, J. R. (2008). Using a model to compute the optimal schedule of practice. *Journal of Experimental Psychology: Applied, 14*(2), 101–117. https://doi.org/10.1037/1076-898X.14.2.101
- Pimsleur, P. (1967). A memory schedule. *The Modern Language Journal, 51*(2), 73–75. https://doi.org/10.1111/j.1540-4781.1967.tb06700.x
- Pyc, M. A., & Rawson, K. A. (2009). Testing the retrieval effort hypothesis: Does greater difficulty correctly recalling information lead to higher levels of memory? *Journal of Memory and Language, 60*(4), 437–447. https://doi.org/10.1016/j.jml.2009.01.004
- Raaijmakers, J. G. W. (2003). Spacing and repetition effects in human memory: Application of the SAM model. *Cognitive Science, 27*(3), 431–452. https://doi.org/10.1207/s15516709cog2703_5
- Rawson, K. A., & Dunlosky, J. (2011). Optimizing schedules of retrieval practice for durable and efficient learning: How much is enough? *Journal of Experimental Psychology: General, 140*(3), 283–302. https://doi.org/10.1037/a0023956
- Reddy, S., Labutov, I., Banerjee, S., & Joachims, T. (2016). Unbounded human learning: Optimal scheduling for spaced repetition. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 1815–1824). https://doi.org/10.1145/2939672.2939850
- Rivera-Lares, K., Baddeley, A., & Della Sala, S. (2025). Influence of degree of learning on rate of forgetting of tonal sequences. *Memory & Cognition, 53*(2), 682–691. https://doi.org/10.3758/s13421-024-01597-6
- Rivera-Lares, K., Della Sala, S., Baddeley, A., & Logie, R. (2023). Rate of forgetting is independent from initial degree of learning across different age groups. *Quarterly Journal of Experimental Psychology, 76*(7), 1672–1682. https://doi.org/10.1177/17470218221128780
- Rivera-Lares, K., Logie, R., Baddeley, A., & Della Sala, S. (2022). Rate of forgetting is independent of initial degree of learning. *Memory & Cognition, 50*(8), 1706–1718. https://doi.org/10.3758/s13421-021-01271-1
- Roediger, H. L., III, & Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science, 17*(3), 249–255. https://doi.org/10.1111/j.1467-9280.2006.01693.x
- Rowland, C. A. (2014). The effect of testing versus restudy on retention: A meta-analytic review of the testing effect. *Psychological Bulletin, 140*(6), 1432–1463. https://doi.org/10.1037/a0037559
- Settles, B., & Meeder, B. (2016). A trainable spaced repetition model for language learning. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1848–1858). https://doi.org/10.18653/v1/P16-1174
- Slamecka, N. J. (1985). On comparing rates of forgetting: Comment on Loftus (1985). *Journal of Experimental Psychology: Learning, Memory, and Cognition, 11*(4), 812–816. https://doi.org/10.1037/0278-7393.11.1-4.812
- Slamecka, N. J., & McElree, B. (1983). Normal forgetting of verbal lists as a function of their degree of learning. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 9*(3), 384–397. https://doi.org/10.1037/0278-7393.9.3.384
- Smolen, P., Zhang, Y., & Byrne, J. H. (2016). The right time to learn: Mechanisms and optimization of spaced learning. *Nature Reviews Neuroscience, 17*(2), 77–88. https://doi.org/10.1038/nrn.2015.18
- Storm, B. C., Bjork, R. A., & Storm, J. C. (2010). Optimizing retrieval as a learning event: When and why expanding retrieval practice enhances long-term retention. *Memory & Cognition, 38*(2), 244–253. https://doi.org/10.3758/MC.38.2.244
- Su, J., Ye, J., Nie, L., Cao, Y., & Chen, Y. (2023). Optimizing spaced repetition schedule by capturing the dynamics of memory. *IEEE Transactions on Knowledge and Data Engineering, 35*(10), 10085–10097. https://doi.org/10.1109/TKDE.2023.3251721
- Tabibian, B., Upadhyay, U., De, A., Zarezade, A., Schölkopf, B., & Gomez-Rodriguez, M. (2019). Enhancing human learning via spaced repetition optimization. *Proceedings of the National Academy of Sciences, 116*(10), 3988–3993. https://doi.org/10.1073/pnas.1815156116
- Thios, S. J., & D'Agostino, P. R. (1976). Effects of repetition as a function of study-phase retrieval. *Journal of Verbal Learning and Verbal Behavior, 15*(5), 529–536. https://doi.org/10.1016/0022-5371(76)90047-5
- Toppino, T. C., & Gerbier, E. (2014). About practice: Repetition, spacing, and abstraction. *Psychology of Learning and Motivation, 60*, 113–189. https://doi.org/10.1016/B978-0-12-800090-8.00004-4
- Toppino, T. C., Phelan, H.-A., & Gerbier, E. (2018). Level of initial training moderates the effects of distributing practice over multiple days with expanding, contracting, and uniform schedules: Evidence for study-phase retrieval. *Memory & Cognition, 46*(6), 969–978. https://doi.org/10.3758/s13421-018-0815-7
- Wahlheim, C. N., Maddox, G. B., & Jacoby, L. L. (2014). The role of reminding in the effects of spaced repetitions on cued recall: Sufficient but not necessary. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 40*(1), 94–105. https://doi.org/10.1037/a0034055
- Walsh, M. M., Gluck, K. A., Gunzelmann, G., Jastrzembski, T., Krusmark, M., Myung, J. I., Pitt, M. A., & Zhou, R. (2018). Mechanisms underlying the spacing effect in learning: A comparison of three computational models. *Journal of Experimental Psychology: General, 147*(9), 1325–1348. https://doi.org/10.1037/xge0000416
- Wiseheart, M., Küpper-Tetzel, C. E., Weston, T., Kim, A. S. N., Kapler, I. V., & Foot-Seymour, V. (2019). Enhancing the quality of student learning using distributed practice. In J. Dunlosky & K. A. Rawson (Eds.), *The Cambridge handbook of cognition and education* (pp. 550–584). Cambridge University Press. https://doi.org/10.1017/9781108235631.023
- Wixted, J. T. (2004). The psychology and neuroscience of forgetting. *Annual Review of Psychology, 55*, 235–269. https://doi.org/10.1146/annurev.psych.55.090902.141555
- Wixted, J. T., & Ebbesen, E. B. (1991). On the form of forgetting. *Psychological Science, 2*(6), 409–415. https://doi.org/10.1111/j.1467-9280.1991.tb00175.x
- Wozniak, P. A. (1990). *Optimization of learning* [Master's thesis, University of Technology in Poznań]. SuperMemo 2 algorithm description (web version 1998): https://super-memory.com/english/ol/sm2.htm
- Yan, V. X., Eglington, L. G., & Garcia, M. A. (2020). Learning better, learning more: The benefits of expanded retrieval practice. *Journal of Applied Research in Memory and Cognition, 9*(2), 204–214. https://doi.org/10.1016/j.jarmac.2020.03.002
- Ye, J., Su, J., & Cao, Y. (2022). A stochastic shortest path algorithm for optimizing spaced repetition scheduling. In *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 4381–4390). https://doi.org/10.1145/3534678.3539081

**Items still UNVERIFIED or partially verified:**
- Leitner (1972): the first-edition year is not confirmed by a primary catalogue.
- Pimsleur's specific interval values.
- Bjork & Bjork (1992), chapter details.
- The full editor list of Bjork & Bjork (2011).
- FSRS as a peer-reviewed source.
- Kang (2016): content only from secondary sources.
- Balota et al. (2007): the author-hosted URL returned HTML at the time of checking.
