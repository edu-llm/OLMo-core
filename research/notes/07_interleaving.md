# 07: Blocked vs. interleaved training: human learning science, continual learning, and LM training order

Prepared 2026-10-04 for TEST 02 of the P4 report ("Blocked vs. Interleaved Subject Training"). The design compares a blocked schedule (80 consecutive subject-pure updates per subject, repeated over four counterbalanced sessions) with an interleaved schedule (switching subject every update). Both arms use matched items, tokens, compute and a constant LR in a ~160–200M causal decoder forked from a common checkpoint, and both receive a shared 160-update generic interference phase afterwards. The proposal used four school subjects; the run actually performed used four synthetic digit-manipulation skills (rotate left, rotate right, reverse, swap pairs).

## TL;DR

- **In humans, interleaving usually helps, but not always, and similarity decides it.** The meta-analytic average is g = 0.42 (59 studies, 238 effect sizes). The benefit is largest for confusable categories (paintings g = 0.67) and small for math (g = 0.34). It reverses for word lists (g = −0.39), for highly dissimilar categories and for rule-based categories, where blocking wins (Brunmair & Richter, 2019; Carvalho & Goldstone, 2014; Noh et al., 2016). Human research therefore does **not** predict a benefit for dissimilar, crisply rule-defined skills.
- **Human interleaving is confounded with spacing.** Kang & Pashler (2012) and Birnbaum et al. (2013) separated the two and found discriminative contrast doing most of the work. Foster et al. (2019) found spacing alone explained the math benefit. Any "interleaving" claim has to say which of the two it means.
- **In networks the reasons differ, and the prediction is near-certain.** With in-weight learning, blocked training causes catastrophic forgetting and interleaving fixes it (McCloskey & Cohen, 1989; McClelland et al., 1995). The key paper, Flesch et al. (2018), found that humans learned better *blocked* while networks needed *interleaving*. Russin et al. (2025) got 50% (blocked) vs. 98–100% (interleaved) with in-weight learning, but a *blocking* advantage when the same transformers learn in context.
- **LM evidence points the same way.** Sequential SFT forgets earlier skills and keeps the last one (Dong et al., 2024, LLaMA 7B–33B). Subject-interleaved instruction tuning beat subject-blocked and random order (Lee et al., 2024, LLaMA-2-13B). Sequential curricula failed in a 22M transformer (Bhasin et al., 2024). Dataset-sequential multitask training was worst in RoBERTa-base (Aghajanyan et al., 2021). There are nulls and near-nulls at 3B (Pareja et al., 2025; Foroutan et al., 2025).
- **Novelty.** "Interleaved beats blocked in a 200M LM" is the expected result, not a novel one. No paper was found that sweeps block length at matched compute in a small causal LM, separates interleaving from spacing and recency with explicit controls, and adds a later interference test. That combination is the defensible contribution.
- **Three confounds to design out:**
  1. Recency: the blocked arm's final block is the freshest.
  2. Spacing: interleaving also spaces each subject.
  3. Optimizer momentum: Adam averages gradients over the last ~10 steps, so "single-subject" interleaved batches effectively become mixed updates. That point is our own analysis, not from the literature.

  The constant LR is the right choice, since LR decay changes how much each position counts (Luo et al., 2026).
- **Scale caveats for ≤ a few B.** Order effects in cyclically repeated (blocked) training, such as anticipatory recovery, grow sharply between 160M and 410M (Yang et al., 2024). Critical mixture ratios rise with size from 460M to 3.1B (Gu et al., 2024). As little as 1% pretraining-data injection prevents forgetting from 41M to 1.27B (Bethune et al., 2025). A 160–200M result may not carry over to 1–3B, so a 2–3-point size sweep is the cheapest strong add-on.

## How items were checked

- **Tags**
  - [V-full]: full text (or the relevant full-text passages) was read in this session.
  - [V-abs]: the abstract or publisher/arXiv/Anthology abstract page was read.
  - [V-meta]: only bibliographic metadata (Crossref/publisher listing) was checked.
  - UNVERIFIED: could not be confirmed.
- "Secondary" marks a number taken from a later paper by the same author rather than from the original.
- ⚠ marks ML evidence only from >7B models or only from non-LM networks.
- Sources were publisher/DOI pages, Crossref, PubMed Central, Europe PMC, ERIC author manuscripts, arXiv, OpenReview, the ACL Anthology, PMLR and NeurIPS proceedings. GitHub was not used.
- The session's web-search quota ran out near the end. The last checks used direct fetches of primary pages, so a few 2025–2026 preprints may be missing.
- **Corrections to commonly repeated claims:**
  1. Brunmair & Richter (2019) report math as *small* (g = 0.34), not moderate. The words effect *favours blocking* (g = −0.39). Expository text is non-significant, not negative.
  2. Zulkiply & Burt (2013): interleaving helped *low*-discriminability (similar) categories, while massing helped high-discriminability ones.
  3. Kornell & Bjork's "78%" is two numbers. 78% did better with spacing, *and* 78% judged massing as good or better.
  4. Kang (2016) is a spacing/testing policy paper. Whether it covers interleaving is UNVERIFIED.
  5. The P4 report cites Rolnick et al. (2019) for "temporal data order can change model forgetting". That paper shows that *replay* reduces forgetting in *RL agents*; it does not manipulate order. Dong et al. (2024) or Russin et al. (2025) support the order claim better.

---

## 1. Definitions

| Term | Definition used here | Verifiable source |
|---|---|---|
| **Blocked (massed) practice** | All practice on one category, problem type or skill is completed before moving to the next (AAA BBB CCC). | Rohrer & Taylor (2007); Rohrer (2012); Kornell & Bjork (2008) call it "massed" |
| **Interleaved practice** | Items of different categories or problem types are mixed within a session (ABC BCA CAB), so consecutive items usually differ in kind. | Rohrer (2012); Brunmair & Richter (2019) |
| **Spacing (distributed practice)** | Repetitions of the *same* item or type are separated in time, as opposed to massed back to back. | Cepeda et al. (2006) |
| **Spacing confound** | Interleaving necessarily also spaces each type, so an interleaving benefit may be a spacing benefit. Separating them needs a blocked-but-spaced control (unrelated filler between same-type items) or a "remote interleaving" control. | Rohrer (2012); Kang & Pashler (2012); Birnbaum et al. (2013); Foster et al. (2019) |
| **Discriminative-contrast hypothesis** | Interleaving helps because putting different categories next to each other highlights the features that tell them apart. | Kang & Pashler (2012); Birnbaum et al. (2013) |
| **Attentional-bias (sequential-attention) account** | Interleaving steers attention to *differences between* categories; blocking steers it to *commonalities within* a category. Which helps depends on what the test requires. | Carvalho & Goldstone (2015, 2017) |
| **Strategy-selection account** | In blocked practice the strategy is given by the block, so learners only *execute* it. Interleaving forces them to *choose* the strategy, which is what tests require. | Rohrer et al. (2015, 2020); Taylor & Rohrer (2010) |
| **Contextual interference** | In motor learning, random (interleaved) practice depresses acquisition but improves retention and transfer relative to blocked practice. | Shea & Morgan (1979) [V-meta; results UNVERIFIED here] |
| **Catastrophic forgetting / interference** | Training a shared network on new data overwrites parameters that supported earlier mappings, causing abrupt loss of earlier skills. It is much more severe than human retroactive interference. | McCloskey & Cohen (1989); French (1999) |
| **Interleaved learning (CLS sense)** | Slow learners can absorb new structure without losing old structure only if new items are mixed with old ones. | McClelland et al. (1995) |
| **Multitask / mixed (joint) training** | All tasks are sampled throughout training. *Batch-homogeneous* means each batch holds one task, with tasks alternating; *batch-heterogeneous* means each update mixes tasks. *Dataset-sequential* is blocked training. | Aghajanyan et al. (2021), §5.2 |
| **Replay / rehearsal** | Re-presenting a fraction of earlier-task data while training on new data. | Robins (1995); Rolnick et al. (2019); Ibrahim et al. (2024) |
| **In-context vs. in-weight learning** | Learning from examples in the prompt without weight change, vs. learning stored in the parameters by gradient updates. | Chan et al. (2022); Russin et al. (2025) |
| **Stability gap** | A transient sharp drop in old-task performance right after a task switch, followed by partial recovery. | De Lange et al. (2023) |
| **Anticipatory recovery** | Under fixed cyclic (blocked, repeating) order, loss on a document starts to recover *before* it recurs. | Yang et al. (2024) |

---

## 2. Human interleaving literature

### 2.1 Foundational laboratory studies

**Kornell & Bjork (2008)** [V-full]
- **Citation:** Kornell, N., & Bjork, R. A. (2008). Learning concepts and categories: Is spacing the "enemy of induction"? *Psychological Science, 19*(6), 585–592. https://doi.org/10.1111/j.1467-9280.2008.02127.x
- **Summary:** Undergraduates studied 6 paintings by each of 12 artists, either massed by artist or interleaved. Samples: Exp 1a N = 120, Exp 1b N = 72, Exp 2 N = 80. Classifying new paintings was better after interleaving: first test block .61 vs .35 (d = 0.99) in Exp 1a and .59 vs .36 (d = 1.28) in Exp 1b. "78% of the participants did better with spaced presentations … but 78% … said that massing was as good as or better than spacing."
- **Relation to our study:** The classic induction result. Its mechanism (discriminating confusable styles) is perceptual category learning, not skill retention under interference.

**Rohrer & Taylor (2007)** [V-meta + secondary]
- **Citation:** Rohrer, D., & Taylor, K. (2007). The shuffling of mathematics problems improves learning. *Instructional Science, 35*(6), 481–498. https://doi.org/10.1007/s11251-007-9015-8
- **Summary:** Exp 2: college students learned to compute the volumes of four obscure solids with blocked or mixed practice. Mixed practice was worse during practice but much better on a test one week later: 63% vs. 20%, d = 1.34 (numbers from Rohrer, 2012, and Rohrer et al., 2015).
- **Relation to our study:** This is the "acquisition worse, retention better" pattern. Our immediate audit and post-interference audit correspond to the practice and delayed tests.

**Taylor & Rohrer (2010)** [V-abs + secondary]
- **Citation:** Taylor, K., & Rohrer, D. (2010). The effects of interleaved practice. *Applied Cognitive Psychology, 24*(6), 837–848. https://doi.org/10.1002/acp.1598
- **Summary:** Fourth graders practised four prism-problem types with **spacing equated** across conditions. Interleaving "impaired practice session performance yet doubled scores on a test given one day later." The test scores were 77% vs. 38%, d = 1.21 (from Rohrer, 2012). Errors under blocking were mostly mismatches between problem and procedure.
- **Relation to our study:** An early human study that equated spacing. It supports the strategy-selection account: the benefit is in knowing *which* procedure to apply. In the LM, the task is named in the prompt, so this "selection" demand is mostly removed.

**Kang & Pashler (2012)** [V-abs + secondary]
- **Citation:** Kang, S. H. K., & Pashler, H. (2012). Learning painting styles: Spacing is advantageous when it promotes discriminative contrast. *Applied Cognitive Psychology, 26*(1), 97–103. https://doi.org/10.1002/acp.1801
- **Summary:**
  - Exp 1: temporally spacing same-artist paintings (gaps filled with unrelated material) did no better than massing, and both were worse than interleaving.
  - Exp 2: showing different artists side by side matched interleaving and beat massing.
  - The authors conclude interleaving helps because it "enhances discriminative contrast."
- **Relation to our study:** The cleanest separation of interleaving from spacing. It suggests a "blocked but spaced" control arm, with generic filler between same-subject updates, to isolate the order effect.

**Birnbaum, Kornell, Bjork & Bjork (2013)** [V-abs]
- **Citation:** Birnbaum, M. S., Kornell, N., Bjork, E. L., & Bjork, R. A. (2013). Why interleaving enhances inductive learning: The roles of discrimination and retrieval. *Memory & Cognition, 41*(3), 392–402. https://doi.org/10.3758/s13421-012-0272-7
- **Summary:** Three experiments with bird and butterfly images separated interleaving from spacing. "Temporal spacing was harmful when it interrupted the juxtaposition of interleaved categories," even with total spacing held constant. Spacing still helped when it did not interrupt discrimination. (The kind of filler and the role of retrieval were not confirmed from the abstract.)
- **Relation to our study:** Supports discriminative contrast over pure spacing for humans. A network has no comparable "juxtaposition" mechanism across optimizer steps, other than optimizer momentum.

**Carvalho & Goldstone (2014a)** [V-abs]
- **Citation:** Carvalho, P. F., & Goldstone, R. L. (2014a). Putting category learning in order: Category structure and temporal arrangement affect the benefit of interleaved over blocked study. *Memory & Cognition, 42*(3), 481–495. https://doi.org/10.3758/s13421-013-0371-0
- **Summary:** Interleaving gave better generalization for categories with **high** within- and between-category similarity. **Blocking was better for low-similarity categories.** Interleaving highlights differences between categories; blocking highlights within-category commonalities.
- **Relation to our study:** The key moderator. For dissimilar subjects (arithmetic vs. geometry), human evidence does not predict an interleaving benefit. For confusable digit skills (rotate-left vs. rotate-right), it does.

**Carvalho & Goldstone (2014b)** [V-abs]
- **Citation:** Carvalho, P. F., & Goldstone, R. L. (2014b). Effects of interleaved and blocked study on delayed test of category learning generalization. *Frontiers in Psychology, 5*, Article 936. https://doi.org/10.3389/fpsyg.2014.00936
- **Summary:** The schedule × category-type interaction survived a 24-hour delay. The authors argue the interleaving advantage is "not primarily due to temporal spacing."
- **Relation to our study:** The similarity moderator persists over a delay, which is analogous to our post-interference audit.

**Carvalho & Goldstone (2015)** [V-abs]
- **Citation:** Carvalho, P. F., & Goldstone, R. L. (2015). The benefits of interleaved and blocked study: Different tasks benefit from different schedules of study. *Psychonomic Bulletin & Review, 22*(1), 281–288. https://doi.org/10.3758/s13423-014-0676-4
- **Summary:** Crossed active (classify, then feedback) vs. passive (labelled) study with interleaved vs. blocked order. The best schedule depended on the study task. The abstract links passive study and blocking to attention to commonalities, and active study and interleaving to attention to differences.
- **Relation to our study:** LM training is more like "passive" labelled study (teacher forcing on the correct answer). The attentional account therefore gives no strong a-priori reason to expect interleaving to win.

**Carvalho & Goldstone (2017)** [V-abs]
- **Citation:** Carvalho, P. F., & Goldstone, R. L. (2017). The sequence of study changes what information is attended to, encoded, and remembered during category learning. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 43*(11), 1699–1719. https://doi.org/10.1037/xlm0000406
- **Summary:** Interleaving improves encoding of the features that separate categories; blocking improves encoding of each category's characteristic features. Whether a sequence helps "depends on the match between what is encoded and what is required at test."
- **Relation to our study:** The attentional-bias account. A test that needs *discrimination* (correctly identifying which transform applies) favours interleaving; a test that needs *within-skill fidelity* (long digit strings executed exactly) may not.

**Wahlheim, Dunlosky & Jacoby (2011)** [V-abs]
- **Citation:** Wahlheim, C. N., Dunlosky, J., & Jacoby, L. L. (2011). Spacing enhances the learning of natural concepts: An investigation of mechanisms, metacognition, and aging. *Memory & Cognition, 39*(5), 750–763. https://doi.org/10.3758/s13421-010-0063-y
- **Summary:** For bird families, spaced (interleaved) study beat massed study, and showing birds in pairs increased the benefit. This supports discrimination-based accounts.

**Kornell, Castel, Eich & Bjork (2010)** [V-abs]
- **Citation:** Kornell, N., Castel, A. D., Eich, T. S., & Bjork, R. A. (2010). Spacing as the friend of both memory and induction in young and older adults. *Psychology and Aging, 25*(2), 498–503. https://doi.org/10.1037/a0017807
- **Summary:** Spacing helped both repetition learning and induction in younger and older adults, even though participants judged massing better.

**Shea & Morgan (1979)** [V-meta]
- **Citation:** Shea, J. B., & Morgan, R. L. (1979). Contextual interference effects on the acquisition, retention, and transfer of a motor skill. *Journal of Experimental Psychology: Human Learning and Memory, 5*(2), 179–187. https://doi.org/10.1037/0278-7393.5.2.179
- **Summary:** The origin of the motor-learning "contextual interference" effect, usually summarized as random practice being worse in acquisition but better at retention and transfer. The abstract was not accessible, so these results are UNVERIFIED here.

### 2.2 Classroom mathematics and science

**Rohrer (2012)** [V-full]
- **Citation:** Rohrer, D. (2012). Interleaving helps students distinguish among similar concepts. *Educational Psychology Review, 24*(3), 355–367. https://doi.org/10.1007/s10648-012-9201-3
- **Summary:** A review that separates interleaving from spacing. It argues that the "most parsimonious explanation" is that interleaving lets learners compare across, and discriminate among, similar kinds of problems.
- **Relation to our study:** The title states the boundary condition: *similar* concepts.

**Rohrer, Dedrick & Burgess (2014)** [V-full]
- **Citation:** Rohrer, D., Dedrick, R. F., & Burgess, K. (2014). The benefit of interleaved mathematics practice is not limited to superficially similar kinds of problems. *Psychonomic Bulletin & Review, 21*(5), 1323–1330. https://doi.org/10.3758/s13423-014-0588-3
- **Summary:** Seventh graders (n = 140) scored 72% vs. 38% (d = 1.05) on a test two weeks later, even though the problem types looked dissimilar. Interleaved practice has two features: different kinds are mixed, *and* same-kind problems are spaced.
- **Relation to our study:** A counterweight to the similarity moderator. For math, the strategy-selection and spacing components can produce a benefit even for dissimilar problem types.

**Rohrer, Dedrick & Stershic (2015)** [V-full]
- **Citation:** Rohrer, D., Dedrick, R. F., & Stershic, S. (2015). Interleaved practice improves mathematics learning. *Journal of Educational Psychology, 107*(3), 900–908. https://doi.org/10.1037/edu0000001
- **Summary:** 126 seventh graders practised graph and slope problems for about 3 months. Interleaved vs. blocked on an unannounced test: 80% vs. 64% (d = 0.42) after 1 day and 74% vs. 42% (d = 0.79) after 30 days.
- **Relation to our study:** The benefit *grows with delay*. The analogous LM prediction is a larger arm difference after interference than immediately after training.

**Rohrer, Dedrick, Hartwig & Cheung (2020)** [V-full]
- **Citation:** Rohrer, D., Dedrick, R. F., Hartwig, M. K., & Cheung, C.-N. (2020). A randomized controlled trial of interleaved mathematics practice. *Journal of Educational Psychology, 112*(1), 40–52. https://doi.org/10.1037/edu0000367
- **Summary:** A preregistered cluster RCT: 54 seventh-grade classes, final N = 787, about 4 months of practice. Unannounced test one month later: 61% vs. 38%, d = 0.83. Interleaving "requires students to choose a strategy and not merely execute a strategy."
- **Relation to our study:** The strongest field evidence. Its proposed mechanism (strategy choice) has a weak analogue in a task-cued LM.

**Samani & Pan (2021)** [V-full]
- **Citation:** Samani, J., & Pan, S. C. (2021). Interleaved practice enhances memory and problem-solving ability in undergraduate physics. *npj Science of Learning, 6*, Article 32. https://doi.org/10.1038/s41539-021-00110-x
- **Summary:** Counterbalanced within-subjects design in UCLA introductory physics (about 290 students). Surprise tests: 0.54 vs. 0.43 (d = 0.40) and 0.47 vs. 0.27 (d = 0.91), interleaved vs. blocked. Students rated interleaving harder and thought they learned less from it. Midterms, taken after cramming, showed no difference.

**Sana & Yan (2022)** [V-abs]
- **Citation:** Sana, F., & Yan, V. X. (2022). Interleaving retrieval practice promotes science learning. *Psychological Science, 33*(5), 782–788. https://doi.org/10.1177/09567976211057507
- **Summary:** 155 high-school students took weekly quizzes. On a test one month later, interleaved quizzes scored 63%, blocked 54% and no quiz 47% (interleaved vs. blocked d = 0.35).

**Foster, Mueller, Was, Rawson & Dunlosky (2019)** [V-abs]
- **Citation:** Foster, N. L., Mueller, M. L., Was, C., Rawson, K. A., & Dunlosky, J. (2019). Why does interleaving improve math learning? The contributions of discriminative contrast and distributed practice. *Memory & Cognition, 47*(6), 1088–1101. https://doi.org/10.3758/s13421-019-00918-4
- **Summary:** For volume-of-solids problems, interleaving beat blocking but was **no better than "remote interleaving"**, in which target problems alternated with unrelated math. Remote-interleaved beat remote-blocked. The authors conclude **distributed practice (spacing)** explains the math benefit.
- **Relation to our study:** The strongest human reason to add a "blocked + generic-filler" (remote-interleaving) arm. If the network benefit is just spacing, that arm will match full interleaving.

### 2.3 Boundary conditions and blocking advantages

- **Carpenter & Mueller (2013)** [V-abs]. Carpenter, S. K., & Mueller, F. E. (2013). The effects of interleaving versus blocking on foreign language pronunciation learning. *Memory & Cognition, 41*(5), 671–682. https://doi.org/10.3758/s13421-012-0291-4. In French spelling-to-sound rule learning, **blocking beat interleaving in all four experiments**.
- **Noh, Yan, Bjork & Maddox (2016)** [V-abs]. Noh, S. M., Yan, V. X., Bjork, R. A., & Maddox, W. T. (2016). Optimal sequencing during category learning: Testing a dual-learning systems perspective. *Cognition, 155*, 23–29. https://doi.org/10.1016/j.cognition.2016.06.007. A crossover: **blocking was better for rule-based categories**, interleaving for information-integration categories. *Our digit transforms are crisp verbalizable rules, the case where humans favour blocking.*
- **Zulkiply & Burt (2013)** [V-abs]. Zulkiply, N., & Burt, J. S. (2013). The exemplar interleaving effect in inductive learning: Moderation by the difficulty of category discriminations. *Memory & Cognition, 41*(1), 16–27. https://doi.org/10.3758/s13421-012-0238-9. Interleaving helped low-discriminability categories; "massing is more effective for high-discriminability categories."
- **Sorensen & Woltz (2016)** [V-abs]. Sorensen, L. J., & Woltz, D. J. (2016). Blocking as a friend of induction in verbal category learning. *Memory & Cognition, 44*(7), 1000–1013. https://doi.org/10.3758/s13421-016-0615-x. With dissimilar verbal examples, starting blocked improved concept learning.
- **Pan, Tajran, Lovelett, Osuna & Rickard (2019)** [V-abs]. Pan, S. C., Tajran, J., Lovelett, J., Osuna, J., & Rickard, T. C. (2019). Does interleaved practice enhance foreign language learning? The effects of training schedule on Spanish verb conjugation skills. *Journal of Educational Psychology, 111*(7), 1172–1188. https://doi.org/10.1037/edu0000336. Within one session, blocking was equal or numerically better. Across two sessions, interleaving was "substantially better" at one week.
- **Yan & Sana (2021)** [V-abs]. Yan, V. X., & Sana, F. (2021). Does the interleaving effect extend to unrelated concepts? *Journal of Educational Psychology, 113*(1), 125–137. https://doi.org/10.1037/edu0000470. The best schedule interleaved at the concept level or the domain level, "but not both or neither."
- **Rowlandson & Simpson (2025)** [V-abs]. Rowlandson, P., & Simpson, A. (2025). Interleaving in mathematical category learning. *Journal for Research in Mathematics Education, 56*(3), 125–147. https://doi.org/10.5951/jresematheduc-2024-0055. Two large classroom experiments on angle relations found a "repeated failure to detect an interleaving-versus-blocking effect." The authors criticize conflating interleaving with spacing.
- **Nemeth & Lipowsky (2024)** [V-abs]. Nemeth, L., & Lipowsky, F. (2024). The role of prior knowledge and need for cognition for the effectiveness of interleaved and blocked practice. *European Journal of Psychology of Education, 39*(2), 907–929. https://doi.org/10.1007/s10212-023-00723-3. 236 third graders learned subtraction strategies. Prior knowledge helped only under blocking, but the condition difference in moderation was not significant.
- **Fuchs et al. (2023)** [V-abs; volume/DOI not checked]. Fuchs, L. S., Malone, A. S., Preacher, K. J., Cho, E., Fuchs, D., & Changas, P. (2023). Next-generation fraction intervention and the long-term advantage of interleaved instruction. *Exceptional Children*. ERIC EJ1372215. https://eric.ed.gov/?id=EJ1372215. Interleaved vs. blocked: g = 0.28 (n.s.) at posttest and g = 0.65 at one-year follow-up.

### 2.4 Reviews and meta-analyses

**Brunmair & Richter (2019)** [V-abs]
- **Citation:** Brunmair, M., & Richter, T. (2019). Similarity matters: A meta-analysis of interleaved learning and its moderators. *Psychological Bulletin, 145*(11), 1029–1052. https://doi.org/10.1037/bul0000209
- **Summary:** "59 studies with 238 effect sizes nested in 158 samples"; overall g = 0.42.
  - Paintings g = 0.67; math g = 0.34 ("small"); expository text and tastes ambiguous and non-significant; **words g = −0.39 (favouring blocking)**.
  - Interleaving helps more when categories are *more similar to each other*, when items *within* a category are *less* similar, and when material is more complex.
  - The moderators best fit the attentional-bias account.
- **Relation to our study:** The P4 report's "g ≈ 0.42" is correct. It should add that the effect is moderated by similarity and is negative for some materials.

**Firth, Rivers & Boyle (2021)** [V-abs]
- **Citation:** Firth, J., Rivers, I., & Boyle, J. (2021). A systematic review of interleaving as a concept learning strategy. *Review of Education, 9*(2), 642–684. https://doi.org/10.1002/rev3.3266
- **Summary:** 26 studies were included, 17 (32 datasets) in the meta-analysis. Benefits reached g = 0.65 for memory and 0.66 for transfer, largest when differences between items are subtle. Most studies were laboratory studies with undergraduates.

**Dunlosky, Rawson, Marsh, Nathan & Willingham (2013)** [V-abs]
- **Citation:** Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham, D. T. (2013). Improving students' learning with effective learning techniques: Promising directions from cognitive and educational psychology. *Psychological Science in the Public Interest, 14*(1), 4–58. https://doi.org/10.1177/1529100612453266
- **Summary:** "Elaborative interrogation, self-explanation, and interleaved practice received moderate utility assessments." Practice testing and distributed practice were rated high utility.
- **Relation to our study:** Interleaving is a less established human principle than spacing or testing. That is reason to treat a null or reversal as plausible.

**Chen, Paas & Sweller (2021)** [V-abs]
- **Citation:** Chen, O., Paas, F., & Sweller, J. (2021). Spacing and interleaving effects require distinct theoretical bases: A systematic review testing the cognitive load and discriminative-contrast hypotheses. *Educational Psychology Review, 33*(4), 1499–1522. https://doi.org/10.1007/s10648-021-09613-w
- **Summary:** Treats spacing as a cognitive-load (resource-recovery) effect and interleaving as discriminative contrast, so the two are distinct effects.

**Kang (2016)** [V-abs; interleaving coverage UNVERIFIED]
- **Citation:** Kang, S. H. K. (2016). Spaced repetition promotes efficient and effective learning: Policy implications for instruction. *Policy Insights from the Behavioral and Brain Sciences, 3*(1), 12–19. https://doi.org/10.1177/2372732215624708
- **Summary:** A policy review of spacing (with testing) for memory, problem solving and transfer. The abstract does not mention interleaving, so cite it only for spacing.

**Yan & Sana (2021b)** [V-meta]. Yan, V. X., & Sana, F. (2021). The robustness of the interleaving benefit. *Journal of Applied Research in Memory and Cognition, 10*(4), 589–602. https://doi.org/10.1016/j.jarmac.2021.05.002. Abstract not seen.

### 2.5 Theoretical accounts and what each predicts for a neural network

| Account | Where stated | Human prediction | Prediction for a gradient-trained LM (our analysis) |
|---|---|---|---|
| Discriminative contrast | Kang & Pashler (2012); Birnbaum et al. (2013); Rohrer (2012) | Interleaving helps when categories are confusable | Weak. Each optimizer step uses only the current batch; the only cross-step "juxtaposition" comes from Adam's momentum (~10-step window at β₁ = 0.9). The prediction is strongest for confusable pairs (rotate-left vs. rotate-right) *if* the test requires discriminating them. |
| Spacing / distributed practice | Foster et al. (2019); Rohrer et al. (2014) | Benefit comes from the lag between same-type items | Testable directly with a blocked-but-spaced (remote-interleaving) arm. In networks, spacing effects are mixed (see research/notes/02–03). |
| Strategy selection / retrieval | Rohrer et al. (2015, 2020); Taylor & Rohrer (2010) | Interleaving trains choosing the procedure | Mostly absent, because the prompt names the skill. It would reappear if the skill cue were removed or made implicit. |
| Attentional bias | Carvalho & Goldstone (2014a, 2015, 2017); Brunmair & Richter (2019) | Interleave similar categories, block dissimilar ones | Analogous to gradient interference. Similar skills share features, so blocked training overwrites them, while dissimilar skills interfere less (Lee et al., 2021; Holton et al., 2025). The prediction runs in the *same direction* as the human one, but for a different reason. |
| Dual systems | Noh et al. (2016); Russin et al. (2025) | Block rule-based tasks, interleave similarity-based ones | Russin et al. map rule learning onto in-context learning (blocking advantage) and incremental learning onto in-weight learning (interleaving advantage). Our test measures in-weight learning, which predicts an interleaving advantage. |
| CLS / catastrophic interference | McClelland et al. (1995); McCloskey & Cohen (1989) | (Not a human-practice theory) | Strong: blocked training overwrites earlier skills and interleaving prevents it. |

---

## 3. Machine-learning literature

### 3.1 Catastrophic interference and the case for interleaved learning

**McCloskey & Cohen (1989)** [V-meta]
- **Citation:** McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), *Psychology of Learning and Motivation* (Vol. 24, pp. 109–165). Academic Press. https://doi.org/10.1016/S0079-7421(08)60536-8
- **Model scale:** ⚠ Small multilayer back-propagation networks (not LMs).
- **Summary:** Training a network on a second set of associations after a first set (sequential, i.e., blocked) can erase the first set. This loss is far more severe than human retroactive interference. Details of the individual simulations (e.g., the arithmetic-facts experiment) come from secondary sources and were not re-read.
- **Relation to our study:** The founding result behind the prior that interleaving will beat blocking in a shared network.

**Ratcliff (1990)** [V-meta]
- **Citation:** Ratcliff, R. (1990). Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions. *Psychological Review, 97*(2), 285–308. https://doi.org/10.1037/0033-295X.97.2.285
- **Model scale:** ⚠ Small back-propagation networks.
- **Summary:** A standard second citation showing that sequential learning in connectionist networks produces forgetting far beyond human forgetting functions.

**French (1999)** [V-meta]
- **Citation:** French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2
- **Model scale:** Review.
- **Summary:** Reviews why overlapping distributed representations cause catastrophic forgetting, and the remedies: rehearsal, sparser representations and dual-memory systems.

**McClelland, McNaughton & O'Reilly (1995)** [V-meta]
- **Citation:** McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review, 102*(3), 419–457. https://doi.org/10.1037/0033-295X.102.3.419
- **Model scale:** Small connectionist models (theory).
- **Summary:** Slow cortical learning can integrate new structure without destroying old structure only through *interleaved* learning. The hippocampus is proposed to supply that interleaving by reinstating memories.
- **Relation to our study:** The network-side theory for why interleaving should help. The mechanism is *avoiding interference*, not discriminative contrast, so a positive LM result does not validate the human theory.

**Kumaran, Hassabis & McClelland (2016)** [V-meta]
- **Citation:** Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. *Trends in Cognitive Sciences, 20*(7), 512–534. https://doi.org/10.1016/j.tics.2016.05.004
- **Model scale:** Review.
- **Summary:** Replay can re-weight experience by goal relevance, and schema-consistent information can be integrated quickly. The theory is linked to experience replay in deep RL.

**Robins (1995)** [V-meta here; V-pub in research/notes/03]
- **Citation:** Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318
- **Model scale:** ⚠ Small back-propagation networks.
- **Summary:** Rehearsing old items while learning new ones, including "sweep" rehearsal of a changing random subset, sharply reduces forgetting. Also introduces pseudo-rehearsal.

**Kirkpatrick et al. (2017)** [V-meta]
- **Citation:** Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences, 114*(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114
- **Model scale:** ⚠ MLPs and deep RL agents.
- **Summary:** Elastic weight consolidation (EWC) penalizes changes to weights that mattered for earlier tasks. It is the standard non-replay alternative to interleaving.

**Rolnick, Ahuja, Schwarz, Lillicrap & Wayne (2019)** [V-abs]
- **Citation:** Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html
- **Model scale:** ⚠ RL agents (Atari, DMLab); not LMs.
- **Summary:** CLEAR, which combines replay with behavioural cloning, greatly reduces forgetting without task boundaries. A small buffer nearly matches an unlimited one.
- **Relation to our study:** Cited in the P4 report. It supports "mixing old data reduces forgetting", not "order alone matters", and it is not LM evidence.

### 3.2 Humans vs. networks on blocked vs. interleaved training (the crucial contrast)

**Flesch, Balaguer, Dekker, Nili & Summerfield (2018)** [V-full]
- **Citation:** Flesch, T., Balaguer, J., Dekker, R., Nili, H., & Summerfield, C. (2018). Comparing continual task learning in minds and machines. *Proceedings of the National Academy of Sciences, 115*(44), E10313–E10322. https://doi.org/10.1073/pnas.1800755115
- **Model scale:** ⚠ Feed-forward CNNs ("several convolutional and fully connected layers") on tree images; online training, 10,000 trials per task, 20 runs. One variant uses a frozen β-VAE embedding.
- **Summary:**
  - Humans (583 included after exclusions) sorting trees by one of two orthogonal rules did better on a later *interleaved* test after **blocked** training. Long blocks (200 trials) beat interleaved training (d = 0.47); blocks of 2 behaved like interleaved training.
  - Blocking produced "factorized" representations in humans.
  - Networks reached ceiling under interleaving but **fell to chance on task 1 after blocked training on task 2**. Interference was localized in the deeper layers.
  - Unsupervised pre-learning of the stimulus space reduced, but did not remove, forgetting.
- **Relation to our study:** The canonical demonstration that the schedule effect *reverses* between humans and standard networks. The team's interleaving hypothesis therefore rests on the network literature, not on the human literature. It also suggests a block-length sweep (B200/B20/B2 → our 80/20/5/1).

**Flesch, Nagy, Saxe & Summerfield (2023)** [V-abs]
- **Citation:** Flesch, T., Nagy, D. G., Saxe, A., & Summerfield, C. (2023). Modelling continual learning in humans with Hebbian context gating and exponentially decaying task signals. *PLOS Computational Biology, 19*(1), e1010808. https://doi.org/10.1371/journal.pcbi.1010808
- **Model scale:** ⚠ Small feed-forward networks.
- **Summary:** "Sluggish" task units and a Hebbian gating step let a network learn two tasks in sequence without forgetting, and reproduce humans' *cost* of interleaved training.
- **Relation to our study:** Shows what a network needs in order to behave like a human (task gating with slow context). Standard transformers lack it, so they are expected to favour interleaving.

**Flesch, Saxe & Summerfield (2023)** [V-meta]
- **Citation:** Flesch, T., Saxe, A., & Summerfield, C. (2023). Continual task learning in natural and artificial agents. *Trends in Neurosciences, 46*(3), 199–210. https://doi.org/10.1016/j.tins.2022.12.006
- **Model scale:** Review.
- **Summary:** Reviews how brains and networks differ in continual task learning. Background citation for the human–network contrast.

**Dekker, Otto & Summerfield (2022)** [V-abs]
- **Citation:** Dekker, R. B., Otto, F., & Summerfield, C. (2022). Curriculum learning for human compositional generalization. *Proceedings of the National Academy of Sciences, 119*(41), e2205582119. https://doi.org/10.1073/pnas.2205582119
- **Model scale:** ⚠ A Hebbian-gating network model (not an LM).
- **Summary:** Human compositional generalization benefits from training "in which items are axis aligned and temporally correlated", i.e., blocked training.

**Russin, Pavlick & Frank (2025)** [V-full]
- **Citation:** Russin, J., Pavlick, E., & Frank, M. J. (2025). Parallel trade-offs in human cognition and neural networks: The dynamic interplay between in-context and in-weight learning. *Proceedings of the National Academy of Sciences, 122*(35), e2510270122. https://doi.org/10.1073/pnas.2510270122 (preprint titled "Human curriculum effects emerge with in-context learning in neural networks", arXiv:2402.08674)
- **Model scale:** Metalearned transformers: 4 layers, 8 heads, d = 64 (category task); 12 layers, d = 64 (compositional task). In-context probes of GPT-3.5-turbo-instruct and Llama 2 (~70B).
- **Summary:**
  - Networks with **in-weight learning only** (trained from scratch) always favoured interleaving, because blocking caused catastrophic forgetting. Category task: 50% blocked vs. 100% (rule-like) and 97.9% (rotated) interleaved.
  - With **in-context learning**, blocking helped: rule-like 99.1% vs. 88.9%; rotated 70.9% vs. 62.8%.
  - GPT-3.5 and Llama 2 also showed a blocking advantage *in context* on the rule-like compositional task.
  - When in-context learning fails, errors drive weight updates and the interleaving advantage returns.
- **Relation to our study:** Probably the most important single citation for framing. Our design measures **in-weight** learning, where interleaving is expected to win. A blocked arm could look better on few-shot *in-context* probes of the same skills. Including both evaluation modes would be new.

**Pesnot Lerousseau & Summerfield (2026)** [V-abs]
- **Citation:** Pesnot Lerousseau, J., & Summerfield, C. (2026). Shared sensitivity to data distribution during learning in humans and transformer networks. *Nature Human Behaviour, 10*(3), 601–614. https://doi.org/10.1038/s41562-025-02359-3
- **Model scale:** ⚠ Minimal transformer (two attention-only layers, one head each), trained from scratch; not an LM.
- **Summary:** Humans (n = 530) and transformers show the same trade-off: redundancy drives in-weight learning and diversity drives in-context learning. But "only people benefit from curricula."
- **Relation to our study:** Recent evidence that human curriculum (ordering) benefits often do not transfer to transformers.

**Holton, Braun, Thompson, Grohn & Summerfield (2025)** [V-full]
- **Citation:** Holton, E., Braun, L., Thompson, J. A. F., Grohn, J., & Summerfield, C. (2025). Humans and neural networks show similar patterns of transfer and interference during continual learning. *Nature Human Behaviour, 10*(1), 111–125. https://doi.org/10.1038/s41562-025-02318-y
- **Model scale:** ⚠ Two-layer linear networks (50 hidden units), with ReLU replications.
- **Summary:**
  - In A→B→A rule learning, both humans and networks transfer more and interfere more when the rules are similar.
  - "Near" tasks caused complete overwriting in rich-regime networks; "far" tasks caused none.
  - Human "lumpers" vs. "splitters" mirror rich vs. lazy network regimes.
  - The paper does not compare blocked vs. interleaved schedules.
- **Relation to our study:** Predicts that similar skills will show the largest blocked-arm forgetting, and the largest interleaving benefit. Rotate-left/right are near tasks: same input, related outputs.

**Beukers, Collin, Kempner, Franklin, Gershman & Norman (2024)** [V-abs]
- **Citation:** Beukers, A. O., Collin, S. H. P., Kempner, R. P., Franklin, N. T., Gershman, S. J., & Norman, K. A. (2024). Blocked training facilitates learning of multiple schemas. *Communications Psychology, 2*, Article 28. https://doi.org/10.1038/s44271-024-00079-4
- **Model scale:** ⚠ Bayesian latent-cause model vs. non-splitting neural network models.
- **Summary:** Humans learned multiple schemas better when blocked than when interleaved. Non-splitting neural models predict the opposite, because of interference.

### 3.3 Task similarity, task order and forgetting (theory and small networks)

**Lee, Goldt & Saxe (2021)** [V-abs]
- **Citation:** Lee, S., Goldt, S., & Saxe, A. (2021). Continual learning in the teacher-student setup: Impact of task similarity. In *Proceedings of the 38th International Conference on Machine Learning* (PMLR Vol. 139, pp. 6109–6119). https://proceedings.mlr.press/v139/lee21e.html
- **Model scale:** ⚠ Two-layer teacher-student networks (analytical).
- **Summary:** "Intermediate task similarity leads to greatest forgetting." Feature similarity and readout similarity have separable effects.

**Ramasesh, Dyer & Raghu (2021)** [V-abs]
- **Citation:** Ramasesh, V. V., Dyer, E., & Raghu, M. (2021). Anatomy of catastrophic forgetting: Hidden representations and task semantics. In *International Conference on Learning Representations (ICLR 2021)*. https://openreview.net/forum?id=LhY8QdUGSuw
- **Model scale:** ⚠ Vision CNNs/ResNets (split CIFAR-10, CIFAR-100 variants).
- **Summary:** Deeper layers are disproportionately the source of forgetting. Forgetting is maximal for task sequences of intermediate similarity.
- **Relation to our study (both papers):** The four digit skills share the input distribution exactly but need different outputs, i.e., shared features with conflicting readouts. Theory therefore predicts substantial blocked-arm forgetting. Rotate-left/right are exact inverses; reverse and swap-pairs are partially overlapping permutations.

**Evron, Moroshko, Ward, Srebro & Soudry (2022)** [V-abs]
- **Citation:** Evron, I., Moroshko, E., Ward, R., Srebro, N., & Soudry, D. (2022). How catastrophic can catastrophic forgetting be in linear regression? In *Proceedings of the Thirty Fifth Conference on Learning Theory* (PMLR Vol. 178). https://arxiv.org/abs/2205.09588
- **Model scale:** ⚠ Overparameterized linear regression (theory).
- **Summary:** Under **cyclic** task order (T tasks repeated k times), forgetting is bounded by T²·min{1/√k, d/k}. Random task order removes the T² factor.
- **Relation to our study:** The blocked arm repeats A→B→C→D four times, which is a cyclic order. Theory says forgetting under cyclic blocking shrinks with repetitions but depends on the number of tasks.

**Li & Hiratani (2025)** [V-abs]
- **Citation:** Li, Z., & Hiratani, N. (2025). Optimal task order for continual learning of multiple tasks. In *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*. https://arxiv.org/abs/2502.03350
- **Model scale:** ⚠ Linear teacher-student theory; MLPs and CNNs on Fashion-MNIST and CIFAR.
- **Summary:** Error after continual learning always depends on task order. Two rules: go "from the least representative to the most typical", and make adjacent tasks dissimilar.
- **Relation to our study:** The counterbalanced orders are not equivalent. Placing rotate-left next to rotate-right is a different condition from separating them. Order manifests should be balanced for adjacency, not just serial position, or order should be analysed as a factor.

### 3.4 Blocked/sequential vs. interleaved/mixed training in transformers and LMs

**Aghajanyan et al. (2021), Muppet** [V-full]
- **Citation:** Aghajanyan, A., Gupta, A., Shrivastava, A., Chen, X., Zettlemoyer, L., & Gupta, S. (2021). Muppet: Massive multi-task representations with pre-finetuning. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing* (pp. 5799–5811). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.emnlp-main.468
- **Model scale:** RoBERTa and BART (base and large). The batching ablation uses **RoBERTa-Base** (~125M) on 35 tasks.
- **Summary:** §5.2 compares three schemes:
  - "dataset homogeneous": train on dataset A, then B, etc. (= **blocked**);
  - "batch homogeneous": each batch is one task, with batches drawn in random order (= **the team's interleaved arm**);
  - "batch heterogeneous": each update mixes tasks.

  Heterogeneous batches "outperform other batching strategies by a significant margin" on downstream fine-tuning. The paper also notes that sequential training of datasets degrades performance. (Exact numbers are only in Figure 2.)
- **Relation to our study:** The closest-scale direct precedent for the three-way distinction. The team's choice of single-subject batches isolates *temporal* order, which is right for the question. A batch-heterogeneous third arm would show how much of the "interleaving" benefit comes from within-update gradient averaging.

**Dong et al. (2024)** [V-full, per sub-agent full-text check]
- **Citation:** Dong, G., Yuan, H., Lu, K., Li, C., Xue, M., Liu, D., Wang, W., Yuan, Z., Zhou, C., & Zhou, J. (2024). How abilities in large language models are affected by supervised fine-tuning data composition. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 177–198). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.acl-long.12
- **Model scale:** ⚠ LLaMA 7B, 13B, 33B.
- **Summary:**
  - Compares multi-task mixing, sequential training (code → math → general), mixed-sequential, and dual-stage mixed fine-tuning (DMT).
  - LLaMA-7B GSM8K: multi-task 47.53 vs. sequential 31.39. LLaMA-33B: 56.69 vs. 47.27.
  - "Sequential training results in catastrophic forgetting," while multi-task mixing produces conflicts. The last-trained skill is preserved.
- **Relation to our study:** Clean LM evidence for forgetting and last-block dominance under blocking, but at ≥7B and in SFT.

**Lee, Cho & Yoo (2024), CORGI** [V-full]
- **Citation:** Lee, B. W., Cho, H., & Yoo, K. M. (2024). Instruction tuning with human curriculum. In *Findings of the Association for Computational Linguistics: NAACL 2024* (pp. 1281–1309). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.findings-naacl.82
- **Model scale:** ⚠ LLaMA 2 13B (instruction tuning).
- **Summary:** Explicitly contrasts "blocking", which "naively stacks hierarchical blocks per subject", with "interleaving", which "cyclically revisits each subject". The paper motivates interleaving with Taylor & Rohrer (2010) and catastrophic forgetting.

  | Order | MMLU | ARC |
  |---|---|---|
  | Interleaved | 57.74 | 58.70 |
  | Blocking | 55.63 | 56.57 |
  | Random shuffle | 54.76 | 57.42 |

  (Table 2, LLaMA 2 13B.) Interleaving is combined with a global easy-to-hard (Bloom's taxonomy) progression, so it is not a pure order manipulation.
- **Relation to our study:** The closest published "human-interleaving-inspired" LM experiment, and the main novelty threat for a framing such as "we test whether interleaving transfers to LMs." The team's design is cleaner (pure order, matched items, interference test, from-checkpoint pairs, ~200M), and that difference should be stated explicitly.

**Bhasin, Ossowski, Zhong & Hu (2024)** [V-full, per sub-agent full-text check; abstract re-checked]
- **Citation:** Bhasin, H., Ossowski, T., Zhong, Y., & Hu, J. (2024). How does multi-task training affect transformer in-context capabilities? Investigations with function classes. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)* (pp. 169–187). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.naacl-short.15
- **Model scale:** GPT-2-style, ~22.4M parameters (8 layers, d = 256).
- **Summary:** For linear, quadratic and cubic function-class tasks:
  - a sequential curriculum (thirds) was "unable to learn any of the tasks";
  - a mixed curriculum (new class plus all earlier classes) was best;
  - fully random mixing failed on quadratic.
- **Relation to our study:** A small-transformer sequential vs. mixed precedent on synthetic tasks, below our scale.

**Yang et al. (2023), MindLLM** [V-full, per sub-agent full-text check]
- **Citation:** Yang, Y., Sun, H., Li, J., Liu, R., Li, Y., Liu, Y., Gao, Y., & Huang, H. (2023). *MindLLM: Pre-training lightweight large language model from scratch, evaluations and domain applications* (arXiv:2310.15777). arXiv. https://arxiv.org/abs/2310.15777
- **Model scale:** 1.3B/3B family; the size used for this ablation was not identified.
- **Summary:** §4.5 compares category-blocked with shuffled pretraining at equal data:
  - loss jumps at each category switch;
  - blocked 5-shot C-Eval led until ~17k steps, then dropped sharply;
  - the authors recommend shuffling.
- **Relation to our study:** A rare from-scratch blocked vs. shuffled pretraining ablation (preprint).

**Pareja et al. (2025)** [V-abs]
- **Citation:** Pareja, A., Nayak, N. S., Wang, H., Killamsetty, K., Sudalairaj, S., Zhao, W., Han, S., Bhandwaldar, A., Xu, G., Xu, K., Han, L., Inglis, L., & Srivastava, A. (2025). Unveiling the secret recipe: A guide for supervised fine-tuning small LLMs. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2412.13337
- **Model scale:** Four open models of 3B–7B.
- **Summary:** "No significant difference in performance between phased and stacked training strategies"; stacked (all at once) is simpler and more sample-efficient.
- **Relation to our study:** A null for phased vs. mixed at 3–7B. Order effects can vanish when phases are long, diverse and large-scale.

**Foroutan, Teiletche, Tarun & Bosselut (2025)** [V-full, per sub-agent]
- **Citation:** Foroutan, N., Teiletche, P., Tarun, A. K., & Bosselut, A. (2025). *Revisiting multilingual data mixtures in language model pretraining* (arXiv:2510.25947). arXiv. https://arxiv.org/abs/2510.25947
- **Model scale:** 1.1B and 3B.
- **Summary:** Staged language introduction "temporarily increases the loss for previously seen languages", but runs "converge to a similar final loss."
- **Relation to our study:** Transient forgetting without a final difference, which is why the *timing* of evaluation matters.

**Fernando et al. (2024)** [V-full, per sub-agent]
- **Citation:** Fernando, H., Shen, H., Ram, P., Zhou, Y., Samulowitz, H., Baracaldo, N., & Chen, T. (2024). *Understanding forgetting in LLM supervised fine-tuning and preference learning – a convex optimization perspective* (arXiv:2410.15483). arXiv. https://arxiv.org/abs/2410.15483
- **Model scale:** Pythia-1B, OPT-1.3B, Llama-3-8B.
- **Summary:** Sequential SFT→DPO leaves a non-vanishing optimality gap. Fine-grained *alternating* training performs about as well as mixing the objectives and beats sequential training (up to 23% on benchmarks).
- **Relation to our study:** Alternation at fine granularity (our interleaved arm) ≈ mixing; sequential is worse. Evidence at 1B.

**Roque & Velasco (2025)** [V-full, per sub-agent]
- **Citation:** Roque, M. T., & Velasco, D. J. (2025). *Beyond repetition: Text simplification and curriculum learning for data-constrained pretraining* (arXiv:2509.24356). BabyLM Workshop. https://arxiv.org/abs/2509.24356
- **Model scale:** ~124M and ~256M.
- **Summary:** Interleaved vs. staged (simple→complex) ordering. Differences were under 1.5 points: staged was marginally better at 124M and interleaved better at 256M.
- **Relation to our study:** The closest in scale. Ordering effects on general LM ability are small at ~100–300M.

**Pilchen, Fabre, Signe Talla, Perez & Grave (2026)** [V-abs]
- **Citation:** Pilchen, H., Fabre, R., Signe Talla, F., Perez, P., & Grave, E. (2026). *Understanding data temporality impact on large language models pre-training* (arXiv:2605.22769). arXiv. https://arxiv.org/abs/2605.22769
- **Model scale:** 6B.
- **Summary:** Chronologically ordered pretraining matches shuffled training on general benchmarks and holds fresher knowledge, at the cost of relative forgetting of older facts.
- **Relation to our study:** Domain-ordered ("temporal") pretraining yields a recency profile, not an across-the-board loss.

**Lee, Sreenivasan, Lee, Lee & Papailiopoulos (2024)** [V-full (partial), per sub-agent]
- **Citation:** Lee, N., Sreenivasan, K., Lee, J. D., Lee, K., & Papailiopoulos, D. (2024). Teaching arithmetic to small transformers. In *International Conference on Learning Representations (ICLR 2024)*. https://arxiv.org/abs/2307.03381
- **Model scale:** NanoGPT ~10.6M up to GPT-2 124M; GPT-3 fine-tuning ⚠.
- **Summary:** Joint training on several arithmetic operations can improve each one. Sequential fine-tuning from 3-digit to 4-digit addition in plain format destroys earlier accuracy, while detailed scratchpads preserve it.
- **Relation to our study:** Digit-level algorithmic skills in small transformers show both joint-training transfer and format-dependent sequential forgetting.

**Furuta, Minegishi, Iwasawa & Matsuo (2024)** [V-abs]
- **Citation:** Furuta, H., Minegishi, G., Iwasawa, Y., & Matsuo, Y. (2024). Towards empirical interpretation of internal circuits and properties in grokked transformers on modular polynomials. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2402.16726
- **Model scale:** Small transformers (sizes not verified).
- **Summary:** Mixing similar operations produces "co-grokking", where generalization accelerates on all tasks; some other mixtures "may not find optimal solutions."
- **Relation to our study:** Whether joint (interleaved) training helps depends on how similar the operations are.

**Chen et al. (2023), Skill-it** [V-full, per sub-agent]
- **Citation:** Chen, M. F., Roberts, N., Bhatia, K., Wang, J., Zhang, C., Sala, F., & Ré, C. (2023). Skill-it! A data-driven skills framework for understanding and training language models. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2307.14430
- **Model scale:** GPT-Neo 125M and 1.3B; 3B for continual pretraining.
- **Summary:** Skills have prerequisite structure. A mixture of prerequisite skills reaches the target-skill loss in ~36% fewer steps than training on the target alone (LEGO). The online skill-mixture schedule beats random and curriculum sampling.
- **Relation to our study:** Skill-it does **not** compare blocked vs. interleaved; all its schedules are mixtures. Cite it for "skills transfer and ordering by dependency matters", not as an order test.

### 3.5 Cyclic order, recency and data order in pretraining

**Yang, Jones, Mozer & Ren (2024)** [V-full]
- **Citation:** Yang, Y., Jones, M., Mozer, M. C., & Ren, M. (2024). Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training. In *Advances in Neural Information Processing Systems 37*. https://doi.org/10.52202/079017-2621
- **Model scale:** Pythia 160M, 410M, 1B, 1.4B, 2.8B (pretrained and random-init). Needs roughly ≥512 width and ≥8 blocks.
- **Summary:**
  - Fine-tuning documents in a fixed cyclic order (10 steps per document, 5 epochs) causes loss to recover *before* a document recurs.
  - "The sharp increase … from the 160M model to the 410M model indicates that anticipatory recovery is an emergent behavior."
  - Cyclic beats shuffled online loss: 1B 1.03 vs. 1.51; 410M 1.09 vs. 1.34.
- **Relation to our study:** The blocked arm's four repeated A→B→C→D sessions are a cyclic schedule. At ~160–200M little anticipatory recovery is expected, but at ≥410M a cyclic blocked arm could partly close the gap. This is a scale-dependent prediction worth testing.

**Luo et al. (2026)** [V-full]
- **Citation:** Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=T5wkZJqzkz
- **Model scale:** 1.5B parameters, 30B tokens.
- **Summary:** Ordering benefits that are clear under a **constant LR** shrink under WSD/cosine decay, because late data receive tiny steps. Fixes are moderate decay or checkpoint averaging (+1.64% average).
- **Relation to our study:** Justifies the constant LR. Under a decaying LR the blocked arm's last block would be down-weighted, confounding order with LR. It also means effects found under constant LR may *shrink* under production schedules, so a WSD sensitivity check is worth considering.

**Krasheninnikov, Turner & Krueger (2025)** [V-abs]
- **Citation:** Krasheninnikov, D., Turner, R. E., & Krueger, D. (2025). *Fresh in memory: Training-order recency is linearly encoded in language model activations* (arXiv:2509.14223). arXiv. https://arxiv.org/abs/2509.14223
- **Model scale:** Llama-3.2-1B, fine-tuned sequentially on 6 disjoint datasets.
- **Summary:** Linear probes recover the training stage at which an entity was learned (~90% early vs. late), and the model can self-report it (~80%).
- **Relation to our study:** Block order leaves a linearly decodable recency trace. A probe for "which session a skill was last trained" could be a mechanistic secondary analysis.

**Biderman et al. (2023), Pythia** [V-full, per sub-agent; metadata re-checked on PMLR]
- **Citation:** Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). Pythia: A suite for analyzing large language models across training and scaling. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 2397–2430). https://proceedings.mlr.press/v202/biderman23a.html
- **Model scale:** 70M–12B; the memorization analysis shown is ⚠ 12B.
- **Summary:** Memorized sequences are distributed across training batches as a Poisson process: "training order has little impact on memorization."
- **Relation to our study:** A counterpoint for verbatim memorization in i.i.d. web data. It does not test skill blocks.

**Lesci, Meister, Hofmann, Vlachos & Pimentel (2024)** [V-abs]
- **Citation:** Lesci, P., Meister, C., Hofmann, T., Vlachos, A., & Pimentel, T. (2024). Causal estimation of memorisation profiles. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics. https://arxiv.org/abs/2406.04327
- **Model scale:** Pythia suite.
- **Summary:** Memorization "is determined by data order and learning rate", refining the Pythia null.

**Liu, Neubig & Xiong (2025)** [V-full, per sub-agent]
- **Citation:** Liu, E., Neubig, G., & Xiong, C. (2025). *Midtraining bridges pretraining and posttraining distributions* (arXiv:2510.14865). arXiv. https://arxiv.org/abs/2510.14865
- **Model scale:** Pythia-architecture 70M, 160M, 410M, 1B.
- **Summary:** When specialized data is introduced interacts strongly with how much of it is used. Late introduction cannot be offset by a larger weight, consistent with a "plasticity window."
- **Relation to our study:** At ~160M, when a skill block arrives matters, which is the serial-position effect that counterbalancing must absorb.

**De Lange, van de Ven & Tuytelaars (2023)** [V-abs]
- **Citation:** De Lange, M., van de Ven, G. M., & Tuytelaars, T. (2023). Continual evaluation for lifelong learning: Identifying the stability gap. In *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2205.13452
- **Model scale:** ⚠ Vision benchmarks.
- **Summary:** At task switches, old-task accuracy drops sharply and then partly recovers, even with replay.
- **Relation to our study:** Evaluate per step around block boundaries, not only at the end of the curriculum.

**Zheng, Cai, Qiu & Ma (2025)** [V-abs]
- **Citation:** Zheng, J., Cai, X., Qiu, S., & Ma, Q. (2025). Spurious forgetting in continual learning of language models. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2501.13453
- **Summary:** Much apparent forgetting in LMs is loss of task alignment, not loss of knowledge, and it recovers quickly.
- **Relation to our study:** Add a short "relearning savings" probe after the interference phase (a few updates per skill) to separate realignment from true loss.

**Mirzadeh, Farajtabar, Pascanu & Ghasemzadeh (2020)** [V-abs]
- **Citation:** Mirzadeh, S. I., Farajtabar, M., Pascanu, R., & Ghasemzadeh, H. (2020). Understanding the role of training regimes in continual learning. In *Advances in Neural Information Processing Systems 33* (pp. 7308–7320). https://arxiv.org/abs/2006.06958
- **Model scale:** ⚠ Vision MLP/CNN.
- **Summary:** Dropout, LR decay and batch size change the width of minima and therefore the amount of forgetting.
- **Relation to our study:** Batch size and LR settings must be identical across arms; they already are in the design.

**Bell & Lawrence (2022)** [V-abs]
- **Citation:** Bell, S. J., & Lawrence, N. D. (2022). *The effect of task ordering in continual learning* (arXiv:2205.13323). arXiv. https://arxiv.org/abs/2205.13323
- **Model scale:** ⚠ Non-LM.
- **Summary:** Task order significantly changes how much is forgotten, which supports counterbalancing.

### 3.6 Data mixing, mixture ratios and forgetting in continual pretraining (≤ few-B)

**Ibrahim et al. (2024)** [V-full]
- **Citation:** Ibrahim, A., Thérien, B., Gupta, K., Richter, M. L., Anthony, Q., Lesort, T., Belilovsky, E., & Rish, I. (2024). Simple and scalable strategies to continually pre-train large language models. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=DimPeeCxKO
- **Model scale:** 405M and ~10B.
- **Summary:** LR re-warming plus re-decay plus a small replay fraction match full retraining. "Even the lowest tested replay of 1% significantly reduces forgetting." Re-warming alone raised loss even without a distribution shift.
- **Relation to our study:** Supports forking with optimizer state restored and no re-warmup. Also shows how little mixing of old data prevents forgetting, so even short subject revisits in the blocked arm may protect it.

**Béthune et al. (2025)** [V-abs]
- **Citation:** Béthune, L., Grangier, D., Busbridge, D., Gualdoni, E., Cuturi, M., & Ablin, P. (2025). Scaling laws for forgetting during finetuning with pretraining data injection. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR Vol. 267, pp. 4020–4042). https://proceedings.mlr.press/v267/bethune25a.html
- **Model scale:** 41M–1.27B.
- **Summary:** "Injecting as little as 1% of pretraining data in the finetuning data mixture prevents the model from forgetting the pretraining set." Forgetting follows a fitted scaling law in model size and fine-tuning tokens.
- **Relation to our study:** At our scale, tiny interleaved doses are powerful, so a "blocked + 1–5% interleaved review" arm is a natural, sharper comparison than pure blocking.

**Gu, Yang, Ding, Zhao & Tan (2024)** [V-full, per sub-agent]
- **Citation:** Gu, J., Yang, Z., Ding, C., Zhao, R., & Tan, F. (2024). CMR scaling law: Predicting critical mixture ratios for continual pre-training of language models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 16143–16162). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.emnlp-main.903
- **Model scale:** 460M, 940M, 1.6B, 3.1B.
- **Summary:** The critical domain ratio at 20B tokens rises with size: 29.8% (460M) → 47.8% (3.1B).
- **Relation to our study:** Larger models tolerate more skewed (block-like) mixtures before forgetting general ability.

**Scialom, Chakrabarty & Muresan (2022)** [V-full, per sub-agent]
- **Citation:** Scialom, T., Chakrabarty, T., & Muresan, S. (2022). Fine-tuned language models are continual learners. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (pp. 6107–6122). Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.emnlp-main.410
- **Model scale:** T0-3B (main), T0pp 11B ⚠.
- **Summary:** With 1% rehearsal, eight sequential tasks retain "almost 100%" of earlier performance. With 0% rehearsal, forgetting is catastrophic.

**Luo et al. (2025)** [V-abs; full text in research/notes/06]
- **Citation:** Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., & Zhang, Y. (2025). An empirical study of catastrophic forgetting in large language models during continual fine-tuning. *IEEE Transactions on Audio, Speech and Language Processing, 33*, 3776–3786. https://doi.org/10.1109/TASLPRO.2025.3606231
- **Model scale:** BLOOMZ 1.1B–7.1B; mT0 1.2B–3.7B.
- **Summary:** Sequential instruction tuning causes forgetting at every size, and the absolute drop grows with scale.

**Xie et al. (2023), DoReMi** [V-abs]
- **Citation:** Xie, S. M., Pham, H., Dong, X., Du, N., Liu, H., Lu, Y., Liang, P., Le, Q. V., Ma, T., & Yu, A. W. (2023). DoReMi: Optimizing data mixtures speeds up language model pretraining. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2305.10429
- **Model scale:** 280M proxy, 8B main ⚠.
- **Summary:** Group-DRO domain weights improve average one-shot accuracy by 6.5 points and reach baseline 2.6× faster on the Pile.
- **Relation to our study:** Concerns mixture *proportions*, not order. Both arms here have identical proportions.

**Ye et al. (2025), Data mixing laws** [V-abs]
- **Citation:** Ye, J., Liu, P., Sun, T., Zhan, J., Zhou, Y., & Qiu, X. (2025). Data mixing laws: Optimizing data mixtures by predicting language modeling performance. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2403.16952
- **Model scale:** Fitted on small models; validated at 1B / 100B tokens.
- **Summary:** Performance is predictable as a function of mixture proportions. An optimized mixture equals 48% more steps on the default mixture, and the laws predict the critical mixture that avoids forgetting in continual training.

**Chen, Hu, Lourie, Cho & Ré (2025), Aioli** [V-full, per sub-agent]
- **Citation:** Chen, M. F., Hu, M. Y., Lourie, N., Cho, K., & Ré, C. (2025). Aioli: A unified optimization framework for language model data mixing. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2411.05735
- **Model scale:** 160M GPT-style decoder.
- **Summary:** No prior mixing method consistently beat stratified (uniform) sampling at 160M; Aioli did, by 0.27 perplexity on average.
- **Relation to our study:** At our exact scale, a uniform interleaved mix is a strong baseline.

**BabyLM findings (curriculum/order)**
- **Warstadt et al. (2023)** [V-full in research/notes/03]. Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., Mosquera, R., Paranjabe, B., Williams, A., Linzen, T., & Cotterell, R. (2023). Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning* (pp. 1–6). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.conll-babylm.1. Curriculum attempts "were largely unsuccessful."
- **Hu et al. (2024)** [V-abs]. Hu, M. Y., Mueller, A., Ross, C., Williams, A., Linzen, T., Zhuang, C., Cotterell, R., Choshen, L., Warstadt, A., & Wilcox, E. G. (2024). *Findings of the second BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora* (arXiv:2412.05149). https://arxiv.org/abs/2412.05149. Curriculum learning again "did not lead to high scores, on average."
- **Model scale:** Small LMs (≤ a few hundred M) on 10M/100M-word budgets.
- **Relation to our study:** Cognitively inspired ordering has a mostly null record in small LMs. The prior for interleaving should come from forgetting, not from pedagogy.

### 3.7 In-context learning and bursty data

**Chan et al. (2022)** [V-abs]
- **Citation:** Chan, S. C. Y., Santoro, A., Lampinen, A. K., Wang, J. X., Singh, A. K., Richemond, P. H., McClelland, J. L., & Hill, F. (2022). Data distributional properties drive emergent in-context learning in transformers. In *Advances in Neural Information Processing Systems 35* (pp. 18878–18891). https://arxiv.org/abs/2205.05055
- **Model scale:** ⚠ Small transformers on Omniglot sequences (not LMs).
- **Summary:** Burstiness (temporal clustering, a within-sequence form of blocking) and many rare classes drive in-context learning, which trades off against in-weight learning.
- **Relation to our study:** Blocked presentation can change *which* learning mode develops, not only how much is retained.

---

## 4. Methodology points specific to this design

1. **Paired runs from a common checkpoint.** This is correct and important. Data order alone contributes about as much variance as weight initialization in fine-tuning (Dodge et al., 2020). Seed-level variance across pretraining runs is mostly small but has outliers (PolyPythias: 2 of 50 runs; van der Wal et al., 2025). Randomizing more variance sources gives a better estimate of the true effect (Bouthillier et al., 2021).
   - Report every pair.
   - Analyse at the pair level, with order manifest and data seed as factors.
   - Check whether any arm has a loss-spike outlier.
2. **Counterbalancing.** Serial-position balancing (a Latin square over four subjects) is necessary but not enough:
   - Task-order theory says *adjacency* also matters: dissimilar neighbours are better, and least-typical tasks should come first (Li & Hiratani, 2025; Bell & Lawrence, 2022).
   - Rotate-left next to rotate-right is a different condition from separating them.
   - Use a design balanced for first-order carryover (a Williams square), or analyse order as a random effect.
   - With 12 pairs, the order × arm interaction will be underpowered. Say so.
3. **Recency.** In the blocked arm, the subject trained last is freshest at the immediate audit (Dong et al., 2024, show the last-trained skill survives sequential SFT). Counterbalancing spreads this across subjects but does not remove it from the macro mean, because the blocked arm always has *one* very fresh subject and one very stale one.
   - Report per-subject accuracy by serial position.
   - Report worst-subject accuracy (already planned).
   - Consider a variant where both arms end with an identical short mixed consolidation segment, or evaluate after each session.
   - The shared generic interference phase partly equalizes recency, which is a strength of the design.
4. **Spacing.** Interleaving also spaces each subject:
   - lag 4 updates in the interleaved arm;
   - in the blocked arm, 80 massed updates, then a ~240-update lag to the next session.

   Foster et al. (2019) show that in humans spacing alone can explain the "interleaving" effect. Add a **remote-interleaving / blocked-but-spaced** arm: subject-pure updates for one subject alternating with generic-data updates at matched compute. If it matches full interleaving, the effect is a spacing (lag) effect, not a mixing effect. The planned 20- and 5-update block conditions give a dose-response curve; Flesch et al. (2018) found B2 ≈ interleaved for humans.
5. **Single-subject vs. mixed batches.** Single-subject batches correctly isolate temporal order. Muppet (Aghajanyan et al., 2021) shows that batch-heterogeneous updates beat batch-homogeneous ones, which in turn beat dataset-sequential training, so a third, mixed-batch arm would measure how much more gradient averaging adds. **Caveat (our analysis, not from a cited paper):**
   - With AdamW (β₁ = 0.9, β₂ ≈ 0.95–0.999), the first moment averages roughly the last 10 updates. In the interleaved arm, every parameter update is therefore a near-even blend of all four subjects' gradients, functionally close to mixed batches.
   - In the blocked arm, the momentum is subject-pure for ~70 of each 80 updates.
   - The second-moment estimate also differs: subject-specific gradient scales get averaged under interleaving.
   - This should be stated as part of the "schedule package". An SGD-without-momentum sanity check or a β₁ = 0 ablation could separate it.
6. **Learning-rate schedule.** Keep the constant LR. Under decay, position and step size are confounded, and ordering effects shrink (Luo et al., 2026). Fork with the optimizer state restored and no re-warmup, since re-warming alone causes forgetting (Ibrahim et al., 2024). If production training will use WSD/cosine, run one sensitivity pair under that schedule.
7. **Evaluation timing and the kind of forgetting.**
   - Evaluate at block boundaries as well as at the end, because stability gaps are transient (De Lange et al., 2023; Foroutan et al., 2025).
   - Add a few-update relearning probe after interference to separate spurious (alignment) forgetting from knowledge loss (Zheng et al., 2025).
   - Optionally add few-shot in-context probes, where blocked training might *help* (Russin et al., 2025).
8. **Seeds and power.** Twelve pairs is a screening design. Differences of a few points need ≥ 5 independent data or packing seeds per order to be distinguishable from order-by-seed noise. Use the screening variance for the confirmatory power analysis, as the report already plans.

**Methods citations**
- **Dodge et al. (2020)** [V-abs]. Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., & Smith, N. (2020). *Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping* (arXiv:2002.06305). arXiv. https://arxiv.org/abs/2002.06305
  - **Model scale:** BERT, hundreds of fine-tuning runs on 4 GLUE tasks.
  - **Summary:** Data order and weight initialization contribute about equally to out-of-sample variance.
- **Bouthillier et al. (2021)** [V-abs]. Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Mohammadi Sepahvand, N., Raff, E., Madan, K., Voleti, V., Ebrahimi Kahou, S., Michalski, V., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). Accounting for variance in machine learning benchmarks. In *Proceedings of Machine Learning and Systems 3* (pp. 747–769). https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html
  - **Model scale:** ⚠ Five benchmark tasks, mostly non-LM.
  - **Summary:** Data sampling, initialization and hyperparameters "impact markedly" results. Randomizing more variance sources improves comparisons.
- **van der Wal et al. (2025)** [V-abs]. van der Wal, O., Lesci, P., Müller-Eberstein, M., Saphra, N., Schoelkopf, H., Zuidema, W., & Biderman, S. (2025). PolyPythias: Stability and outliers across fifty language model pre-training runs. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2503.09543
  - **Model scale:** Pythia 14M–410M, 10 seeds each.
  - **Summary:** Training is largely stable across seeds, with two outlier runs (410M seeds 3 and 4).

---

## 5. Synthesis

### (a) What human research predicts

- **The average human benefit is real but moderate and heterogeneous.** Overall g = 0.42; math g = 0.34; words g = −0.39 (Brunmair & Richter, 2019). It is rated only "moderate utility" (Dunlosky et al., 2013), and there are recent classroom nulls (Rowlandson & Simpson, 2025).
- **It depends on similarity.** Interleaving helps when categories are similar or confusable and items within a category vary. Blocking is as good or better for dissimilar categories (Carvalho & Goldstone, 2014a; Zulkiply & Burt, 2013) and for explicit rule-based categories (Noh et al., 2016; Carpenter & Mueller, 2013; Flesch et al., 2018; Dekker et al., 2022).
- **For the original subjects** (arithmetic, logic, geometry, statistics), which are dissimilar domains, human research does *not* clearly predict an interleaving benefit. The math-classroom results come from interleaving *problem types within* math, where choosing a strategy is the bottleneck (Rohrer et al., 2015, 2020).
- **For the digit skills**, the picture is mixed:
  - rotate-left vs. rotate-right are highly confusable, so discriminative contrast predicts an interleaving benefit;
  - all four are crisp rules, so dual-system accounts predict a blocking benefit;
  - the task is named in the prompt, so the strategy-selection mechanism is largely switched off.
- **The mechanisms do not carry over.** Human mechanisms (discriminative contrast, strategy selection, attention) have no direct analogue in a single SGD step. The human literature motivates the question; it does not supply a mechanism for the LM. Flesch et al. (2018) is the decisive citation: the same manipulation that helps humans when blocked helps networks when interleaved.

### (b) What ML research predicts, and is a positive result novel?

- **Prediction: interleaving will almost certainly reduce forgetting relative to long blocks** when learning is in the weights:
  - catastrophic interference (McCloskey & Cohen, 1989; McClelland et al., 1995);
  - human-vs-network studies (Flesch et al., 2018; Russin et al., 2025);
  - LM and transformer studies: Aghajanyan et al. (2021, ~125M); Bhasin et al. (2024, 22M); Dong et al. (2024, 7–33B); Lee et al. (2024, 13B); Fernando et al. (2024, 1–8B).

  The four skills share inputs but need conflicting outputs, the intermediate-similarity regime where forgetting is maximal (Lee et al., 2021; Ramasesh et al., 2021; Holton et al., 2025). The team's toy MLP pilot (52.8% vs. 79.2%) is consistent with this.
- **Where the effect could shrink or vanish:**
  1. The blocked arm revisits every subject four times, so cyclic repetition and the small "replay-like" doses already limit forgetting (Evron et al., 2022; Ibrahim et al., 2024; Béthune et al., 2025).
  2. Forgetting after a switch may be largely spurious (fast realignment) rather than knowledge loss (Zheng et al., 2025).
  3. The shared generic interference phase may dominate both arms equally.
  4. At 3–7B, phased and stacked SFT showed no difference (Pareja et al., 2025), and staged multilingual curricula converged to the same loss (Foroutan et al., 2025).
- **Novelty.** "Interleaved > blocked at matched compute" is the expected direction. By itself it is a *replication in a new setting* (from-checkpoint continued pretraining of a ~200M causal LM on synthetic skills), not a discovery. CORGI (Lee et al., 2024) has already published a human-interleaving-motivated LM comparison favouring interleaving, though it is confounded with a difficulty curriculum. A *null* or a *reversal* would be more surprising, but would need careful power analysis.

### (c) What would make the experiment novel and publishable

No paper found does the following in a small causal LM at matched compute. Each item maps onto a human-literature distinction that has never been tested cleanly in LMs.

1. **A block-length dose-response.** Sweep 80 / 20 / 5 / 1 updates (the report already lists this as optional; make it primary). Fit forgetting as a function of run length. Flesch et al. (2018) did this for humans (B200/B20/B2); no LM equivalent was found.
2. **Separating interleaving from spacing.** Add a remote-interleaving arm: subject-pure updates alternating with generic data (Foster et al., 2019; Kang & Pashler, 2012). This is the human field's key control and appears untested in LMs.
3. **Separating interleaving from recency.**
   - Report per-subject accuracy by serial position.
   - Add a variant where both arms end with an identical mixed segment.
   - Model accuracy against "updates since last exposure" across arms. If that single variable explains the arm difference, the effect is recency, not interleaving.
4. **A similarity moderator inside the design.** Compare confusable pairs (rotate-left/right, inverse operations) with less related pairs (reverse vs. swap-pairs).
   - Human discriminative contrast and attentional bias predict a larger interleaving benefit for similar pairs (Brunmair & Richter, 2019; Carvalho & Goldstone, 2014a).
   - Network theory predicts maximal blocked-arm forgetting at *intermediate* similarity (Lee et al., 2021; Ramasesh et al., 2021).

   Where the two predictions diverge is a genuine test.
5. **In-weight vs. in-context evaluation.** Score the same skills both zero-shot (in-weight) and few-shot (in-context). Russin et al. (2025) predict that blocking can help in-context performance while hurting in-weight retention. Showing that in a trained ~200M LM would be new.
6. **Temporal order vs. gradient mixing.** Run three arms: blocked, batch-homogeneous interleaved and batch-heterogeneous (Aghajanyan et al., 2021), optionally with a β₁ = 0 ablation. This separates how much of "interleaving" is optimizer averaging.
7. **A scale sweep.** Run the same protocol at ~160M, ~410M and ~1B (e.g., Pythia or DataDecide checkpoints). Anticipatory recovery under cyclic order emerges between 160M and 410M (Yang et al., 2024), so the blocked arm's handicap may *shrink* with scale.
8. **Reporting the "acquisition worse, retention better" pattern.** Interleaving often looks worse during training but better later (Rohrer & Taylor, 2007; Taylor & Rohrer, 2010; Rohrer et al., 2015). An arm × time-of-test interaction would connect directly to the human literature.

**Recommended framing:** "Which components of the human 'interleaving' package (mixing, spacing, recency, discrimination) carry over to in-weight learning in small LMs, and at what block length?" rather than "Does interleaving help LMs?"

### (d) Scale caveats for ≤ few-B models

- **Most direct LM order comparisons are at ≥7B** (Dong et al., 2024; Lee et al., 2024; Pareja et al., 2025, 3–7B). Below 1B, the evidence is from small synthetic transformers (Bhasin et al., 2024; Russin et al., 2025; Lee et al., 2024 arithmetic) or from multitask fine-tuning of encoders (Aghajanyan et al., 2021). The 1–3B range has few direct tests (Fernando et al., 2024; Foroutan et al., 2025, 1.1–3B).
- **Scale-dependent phenomena fall in the 0.1–3B range:**
  - anticipatory recovery emerges between 160M and 410M (Yang et al., 2024);
  - critical mixture ratios rise from 29.8% at 460M to 47.8% at 3.1B (Gu et al., 2024);
  - 1% replay suffices across 41M–1.27B (Béthune et al., 2025);
  - absolute forgetting during sequential instruction tuning *grows* from 1B to 7B (Luo et al., 2025).

  A 160–200M effect size should not be extrapolated to 1–3B without at least one larger point.
- **Ordering effects on general LM quality are small at 100–300M** (under 1.5 points in Roque & Velasco, 2025). Uniform mixing is a strong baseline at 160M (Chen et al., 2025, Aioli). Effects on narrow synthetic skills may be large while effects on general ability stay small, so keep the two outcomes separate.
- **Constant-LR results may overstate ordering effects** relative to standard decaying schedules (Luo et al., 2026, at 1.5B).
- **Human–network contrasts mostly come from non-LM networks** (Flesch et al., 2018; Holton et al., 2025; Pesnot Lerousseau & Summerfield, 2026) or from in-context probes of very large LLMs (Russin et al., 2025, GPT-3.5 and Llama-2-70B). Our ~200M in-weight setting sits between the two.

---

## 6. References (APA 7)

Aghajanyan, A., Gupta, A., Shrivastava, A., Chen, X., Zettlemoyer, L., & Gupta, S. (2021). Muppet: Massive multi-task representations with pre-finetuning. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing* (pp. 5799–5811). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.emnlp-main.468

Bell, S. J., & Lawrence, N. D. (2022). *The effect of task ordering in continual learning* (arXiv:2205.13323). arXiv. https://arxiv.org/abs/2205.13323

Béthune, L., Grangier, D., Busbridge, D., Gualdoni, E., Cuturi, M., & Ablin, P. (2025). Scaling laws for forgetting during finetuning with pretraining data injection. In *Proceedings of the 42nd International Conference on Machine Learning* (PMLR Vol. 267, pp. 4020–4042). https://proceedings.mlr.press/v267/bethune25a.html

Beukers, A. O., Collin, S. H. P., Kempner, R. P., Franklin, N. T., Gershman, S. J., & Norman, K. A. (2024). Blocked training facilitates learning of multiple schemas. *Communications Psychology, 2*, Article 28. https://doi.org/10.1038/s44271-024-00079-4

Bhasin, H., Ossowski, T., Zhong, Y., & Hu, J. (2024). How does multi-task training affect transformer in-context capabilities? Investigations with function classes. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)* (pp. 169–187). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.naacl-short.15

Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). Pythia: A suite for analyzing large language models across training and scaling. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR Vol. 202, pp. 2397–2430). https://proceedings.mlr.press/v202/biderman23a.html

Birnbaum, M. S., Kornell, N., Bjork, E. L., & Bjork, R. A. (2013). Why interleaving enhances inductive learning: The roles of discrimination and retrieval. *Memory & Cognition, 41*(3), 392–402. https://doi.org/10.3758/s13421-012-0272-7

Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Mohammadi Sepahvand, N., Raff, E., Madan, K., Voleti, V., Ebrahimi Kahou, S., Michalski, V., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). Accounting for variance in machine learning benchmarks. In *Proceedings of Machine Learning and Systems 3* (pp. 747–769). https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html

Brunmair, M., & Richter, T. (2019). Similarity matters: A meta-analysis of interleaved learning and its moderators. *Psychological Bulletin, 145*(11), 1029–1052. https://doi.org/10.1037/bul0000209

Carpenter, S. K., & Mueller, F. E. (2013). The effects of interleaving versus blocking on foreign language pronunciation learning. *Memory & Cognition, 41*(5), 671–682. https://doi.org/10.3758/s13421-012-0291-4

Carvalho, P. F., & Goldstone, R. L. (2014a). Putting category learning in order: Category structure and temporal arrangement affect the benefit of interleaved over blocked study. *Memory & Cognition, 42*(3), 481–495. https://doi.org/10.3758/s13421-013-0371-0

Carvalho, P. F., & Goldstone, R. L. (2014b). Effects of interleaved and blocked study on delayed test of category learning generalization. *Frontiers in Psychology, 5*, Article 936. https://doi.org/10.3389/fpsyg.2014.00936

Carvalho, P. F., & Goldstone, R. L. (2015). The benefits of interleaved and blocked study: Different tasks benefit from different schedules of study. *Psychonomic Bulletin & Review, 22*(1), 281–288. https://doi.org/10.3758/s13423-014-0676-4

Carvalho, P. F., & Goldstone, R. L. (2017). The sequence of study changes what information is attended to, encoded, and remembered during category learning. *Journal of Experimental Psychology: Learning, Memory, and Cognition, 43*(11), 1699–1719. https://doi.org/10.1037/xlm0000406

Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin, 132*(3), 354–380. https://doi.org/10.1037/0033-2909.132.3.354

Chan, S. C. Y., Santoro, A., Lampinen, A. K., Wang, J. X., Singh, A. K., Richemond, P. H., McClelland, J. L., & Hill, F. (2022). Data distributional properties drive emergent in-context learning in transformers. In *Advances in Neural Information Processing Systems 35* (pp. 18878–18891). https://arxiv.org/abs/2205.05055

Chen, M. F., Hu, M. Y., Lourie, N., Cho, K., & Ré, C. (2025). Aioli: A unified optimization framework for language model data mixing. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2411.05735

Chen, M. F., Roberts, N., Bhatia, K., Wang, J., Zhang, C., Sala, F., & Ré, C. (2023). Skill-it! A data-driven skills framework for understanding and training language models. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2307.14430

Chen, O., Paas, F., & Sweller, J. (2021). Spacing and interleaving effects require distinct theoretical bases: A systematic review testing the cognitive load and discriminative-contrast hypotheses. *Educational Psychology Review, 33*(4), 1499–1522. https://doi.org/10.1007/s10648-021-09613-w

De Lange, M., van de Ven, G. M., & Tuytelaars, T. (2023). Continual evaluation for lifelong learning: Identifying the stability gap. In *International Conference on Learning Representations (ICLR 2023)*. https://arxiv.org/abs/2205.13452

Dekker, R. B., Otto, F., & Summerfield, C. (2022). Curriculum learning for human compositional generalization. *Proceedings of the National Academy of Sciences, 119*(41), e2205582119. https://doi.org/10.1073/pnas.2205582119

Dodge, J., Ilharco, G., Schwartz, R., Farhadi, A., Hajishirzi, H., & Smith, N. (2020). *Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping* (arXiv:2002.06305). arXiv. https://arxiv.org/abs/2002.06305

Dong, G., Yuan, H., Lu, K., Li, C., Xue, M., Liu, D., Wang, W., Yuan, Z., Zhou, C., & Zhou, J. (2024). How abilities in large language models are affected by supervised fine-tuning data composition. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 177–198). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.acl-long.12

Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., & Willingham, D. T. (2013). Improving students' learning with effective learning techniques: Promising directions from cognitive and educational psychology. *Psychological Science in the Public Interest, 14*(1), 4–58. https://doi.org/10.1177/1529100612453266

Evron, I., Moroshko, E., Ward, R., Srebro, N., & Soudry, D. (2022). How catastrophic can catastrophic forgetting be in linear regression? In *Proceedings of the Thirty Fifth Conference on Learning Theory* (PMLR Vol. 178). https://arxiv.org/abs/2205.09588

Fernando, H., Shen, H., Ram, P., Zhou, Y., Samulowitz, H., Baracaldo, N., & Chen, T. (2024). *Understanding forgetting in LLM supervised fine-tuning and preference learning – a convex optimization perspective* (arXiv:2410.15483). arXiv. https://arxiv.org/abs/2410.15483

Firth, J., Rivers, I., & Boyle, J. (2021). A systematic review of interleaving as a concept learning strategy. *Review of Education, 9*(2), 642–684. https://doi.org/10.1002/rev3.3266

Flesch, T., Balaguer, J., Dekker, R., Nili, H., & Summerfield, C. (2018). Comparing continual task learning in minds and machines. *Proceedings of the National Academy of Sciences, 115*(44), E10313–E10322. https://doi.org/10.1073/pnas.1800755115

Flesch, T., Nagy, D. G., Saxe, A., & Summerfield, C. (2023). Modelling continual learning in humans with Hebbian context gating and exponentially decaying task signals. *PLOS Computational Biology, 19*(1), e1010808. https://doi.org/10.1371/journal.pcbi.1010808

Flesch, T., Saxe, A., & Summerfield, C. (2023). Continual task learning in natural and artificial agents. *Trends in Neurosciences, 46*(3), 199–210. https://doi.org/10.1016/j.tins.2022.12.006

Foroutan, N., Teiletche, P., Tarun, A. K., & Bosselut, A. (2025). *Revisiting multilingual data mixtures in language model pretraining* (arXiv:2510.25947). arXiv. https://arxiv.org/abs/2510.25947

Foster, N. L., Mueller, M. L., Was, C., Rawson, K. A., & Dunlosky, J. (2019). Why does interleaving improve math learning? The contributions of discriminative contrast and distributed practice. *Memory & Cognition, 47*(6), 1088–1101. https://doi.org/10.3758/s13421-019-00918-4

French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences, 3*(4), 128–135. https://doi.org/10.1016/S1364-6613(99)01294-2

Fuchs, L. S., Malone, A. S., Preacher, K. J., Cho, E., Fuchs, D., & Changas, P. (2023). Next-generation fraction intervention and the long-term advantage of interleaved instruction. *Exceptional Children*. https://eric.ed.gov/?id=EJ1372215 [volume/DOI not checked]

Furuta, H., Minegishi, G., Iwasawa, Y., & Matsuo, Y. (2024). Towards empirical interpretation of internal circuits and properties in grokked transformers on modular polynomials. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2402.16726

Gu, J., Yang, Z., Ding, C., Zhao, R., & Tan, F. (2024). CMR scaling law: Predicting critical mixture ratios for continual pre-training of language models. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 16143–16162). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.emnlp-main.903

Holton, E., Braun, L., Thompson, J. A. F., Grohn, J., & Summerfield, C. (2025). Humans and neural networks show similar patterns of transfer and interference during continual learning. *Nature Human Behaviour, 10*(1), 111–125. https://doi.org/10.1038/s41562-025-02318-y

Hu, M. Y., Mueller, A., Ross, C., Williams, A., Linzen, T., Zhuang, C., Cotterell, R., Choshen, L., Warstadt, A., & Wilcox, E. G. (2024). *Findings of the second BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora* (arXiv:2412.05149). arXiv. https://arxiv.org/abs/2412.05149

Ibrahim, A., Thérien, B., Gupta, K., Richter, M. L., Anthony, Q., Lesort, T., Belilovsky, E., & Rish, I. (2024). Simple and scalable strategies to continually pre-train large language models. *Transactions on Machine Learning Research*. https://openreview.net/forum?id=DimPeeCxKO

Kang, S. H. K. (2016). Spaced repetition promotes efficient and effective learning: Policy implications for instruction. *Policy Insights from the Behavioral and Brain Sciences, 3*(1), 12–19. https://doi.org/10.1177/2372732215624708

Kang, S. H. K., & Pashler, H. (2012). Learning painting styles: Spacing is advantageous when it promotes discriminative contrast. *Applied Cognitive Psychology, 26*(1), 97–103. https://doi.org/10.1002/acp.1801

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences, 114*(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114

Kornell, N., & Bjork, R. A. (2008). Learning concepts and categories: Is spacing the "enemy of induction"? *Psychological Science, 19*(6), 585–592. https://doi.org/10.1111/j.1467-9280.2008.02127.x

Kornell, N., Castel, A. D., Eich, T. S., & Bjork, R. A. (2010). Spacing as the friend of both memory and induction in young and older adults. *Psychology and Aging, 25*(2), 498–503. https://doi.org/10.1037/a0017807

Krasheninnikov, D., Turner, R. E., & Krueger, D. (2025). *Fresh in memory: Training-order recency is linearly encoded in language model activations* (arXiv:2509.14223). arXiv. https://arxiv.org/abs/2509.14223

Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. *Trends in Cognitive Sciences, 20*(7), 512–534. https://doi.org/10.1016/j.tics.2016.05.004

Lee, B. W., Cho, H., & Yoo, K. M. (2024). Instruction tuning with human curriculum. In *Findings of the Association for Computational Linguistics: NAACL 2024* (pp. 1281–1309). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.findings-naacl.82

Lee, N., Sreenivasan, K., Lee, J. D., Lee, K., & Papailiopoulos, D. (2024). Teaching arithmetic to small transformers. In *International Conference on Learning Representations (ICLR 2024)*. https://arxiv.org/abs/2307.03381

Lee, S., Goldt, S., & Saxe, A. (2021). Continual learning in the teacher-student setup: Impact of task similarity. In *Proceedings of the 38th International Conference on Machine Learning* (PMLR Vol. 139, pp. 6109–6119). https://proceedings.mlr.press/v139/lee21e.html

Lesci, P., Meister, C., Hofmann, T., Vlachos, A., & Pimentel, T. (2024). Causal estimation of memorisation profiles. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Association for Computational Linguistics. https://arxiv.org/abs/2406.04327

Li, Z., & Hiratani, N. (2025). Optimal task order for continual learning of multiple tasks. In *Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)*. https://arxiv.org/abs/2502.03350

Liu, E., Neubig, G., & Xiong, C. (2025). *Midtraining bridges pretraining and posttraining distributions* (arXiv:2510.14865). arXiv. https://arxiv.org/abs/2510.14865

Luo, K., Sun, Z., Wen, H., Shi, X., Cui, J., Dang, C., Lyu, K., & Chen, W. (2026). How learning rate decay wastes your best data in curriculum-based LLM pretraining. In *International Conference on Learning Representations (ICLR 2026)*. https://openreview.net/forum?id=T5wkZJqzkz

Luo, Y., Yang, Z., Meng, F., Li, Y., Zhou, J., & Zhang, Y. (2025). An empirical study of catastrophic forgetting in large language models during continual fine-tuning. *IEEE Transactions on Audio, Speech and Language Processing, 33*, 3776–3786. https://doi.org/10.1109/TASLPRO.2025.3606231

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. *Psychological Review, 102*(3), 419–457. https://doi.org/10.1037/0033-295X.102.3.419

McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), *Psychology of Learning and Motivation* (Vol. 24, pp. 109–165). Academic Press. https://doi.org/10.1016/S0079-7421(08)60536-8

Mirzadeh, S. I., Farajtabar, M., Pascanu, R., & Ghasemzadeh, H. (2020). Understanding the role of training regimes in continual learning. In *Advances in Neural Information Processing Systems 33* (pp. 7308–7320). https://arxiv.org/abs/2006.06958

Nemeth, L., & Lipowsky, F. (2024). The role of prior knowledge and need for cognition for the effectiveness of interleaved and blocked practice. *European Journal of Psychology of Education, 39*(2), 907–929. https://doi.org/10.1007/s10212-023-00723-3

Noh, S. M., Yan, V. X., Bjork, R. A., & Maddox, W. T. (2016). Optimal sequencing during category learning: Testing a dual-learning systems perspective. *Cognition, 155*, 23–29. https://doi.org/10.1016/j.cognition.2016.06.007

Pan, S. C., Tajran, J., Lovelett, J., Osuna, J., & Rickard, T. C. (2019). Does interleaved practice enhance foreign language learning? The effects of training schedule on Spanish verb conjugation skills. *Journal of Educational Psychology, 111*(7), 1172–1188. https://doi.org/10.1037/edu0000336

Pareja, A., Nayak, N. S., Wang, H., Killamsetty, K., Sudalairaj, S., Zhao, W., Han, S., Bhandwaldar, A., Xu, G., Xu, K., Han, L., Inglis, L., & Srivastava, A. (2025). Unveiling the secret recipe: A guide for supervised fine-tuning small LLMs. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2412.13337

Pesnot Lerousseau, J., & Summerfield, C. (2026). Shared sensitivity to data distribution during learning in humans and transformer networks. *Nature Human Behaviour, 10*(3), 601–614. https://doi.org/10.1038/s41562-025-02359-3

Pilchen, H., Fabre, R., Signe Talla, F., Perez, P., & Grave, E. (2026). *Understanding data temporality impact on large language models pre-training* (arXiv:2605.22769). arXiv. https://arxiv.org/abs/2605.22769

Ramasesh, V. V., Dyer, E., & Raghu, M. (2021). Anatomy of catastrophic forgetting: Hidden representations and task semantics. In *International Conference on Learning Representations (ICLR 2021)*. https://openreview.net/forum?id=LhY8QdUGSuw

Ratcliff, R. (1990). Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions. *Psychological Review, 97*(2), 285–308. https://doi.org/10.1037/0033-295X.97.2.285

Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science, 7*(2), 123–146. https://doi.org/10.1080/09540099550039318

Rohrer, D. (2012). Interleaving helps students distinguish among similar concepts. *Educational Psychology Review, 24*(3), 355–367. https://doi.org/10.1007/s10648-012-9201-3

Rohrer, D., Dedrick, R. F., & Burgess, K. (2014). The benefit of interleaved mathematics practice is not limited to superficially similar kinds of problems. *Psychonomic Bulletin & Review, 21*(5), 1323–1330. https://doi.org/10.3758/s13423-014-0588-3

Rohrer, D., Dedrick, R. F., Hartwig, M. K., & Cheung, C.-N. (2020). A randomized controlled trial of interleaved mathematics practice. *Journal of Educational Psychology, 112*(1), 40–52. https://doi.org/10.1037/edu0000367

Rohrer, D., Dedrick, R. F., & Stershic, S. (2015). Interleaved practice improves mathematics learning. *Journal of Educational Psychology, 107*(3), 900–908. https://doi.org/10.1037/edu0000001

Rohrer, D., & Taylor, K. (2007). The shuffling of mathematics problems improves learning. *Instructional Science, 35*(6), 481–498. https://doi.org/10.1007/s11251-007-9015-8

Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G. (2019). Experience replay for continual learning. In *Advances in Neural Information Processing Systems 32*. https://proceedings.neurips.cc/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html

Roque, M. T., & Velasco, D. J. (2025). *Beyond repetition: Text simplification and curriculum learning for data-constrained pretraining* (arXiv:2509.24356). arXiv. https://arxiv.org/abs/2509.24356

Rowlandson, P., & Simpson, A. (2025). Interleaving in mathematical category learning. *Journal for Research in Mathematics Education, 56*(3), 125–147. https://doi.org/10.5951/jresematheduc-2024-0055

Russin, J., Pavlick, E., & Frank, M. J. (2025). Parallel trade-offs in human cognition and neural networks: The dynamic interplay between in-context and in-weight learning. *Proceedings of the National Academy of Sciences, 122*(35), e2510270122. https://doi.org/10.1073/pnas.2510270122

Samani, J., & Pan, S. C. (2021). Interleaved practice enhances memory and problem-solving ability in undergraduate physics. *npj Science of Learning, 6*, Article 32. https://doi.org/10.1038/s41539-021-00110-x

Sana, F., & Yan, V. X. (2022). Interleaving retrieval practice promotes science learning. *Psychological Science, 33*(5), 782–788. https://doi.org/10.1177/09567976211057507

Scialom, T., Chakrabarty, T., & Muresan, S. (2022). Fine-tuned language models are continual learners. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (pp. 6107–6122). Association for Computational Linguistics. https://doi.org/10.18653/v1/2022.emnlp-main.410

Shea, J. B., & Morgan, R. L. (1979). Contextual interference effects on the acquisition, retention, and transfer of a motor skill. *Journal of Experimental Psychology: Human Learning and Memory, 5*(2), 179–187. https://doi.org/10.1037/0278-7393.5.2.179

Sorensen, L. J., & Woltz, D. J. (2016). Blocking as a friend of induction in verbal category learning. *Memory & Cognition, 44*(7), 1000–1013. https://doi.org/10.3758/s13421-016-0615-x

Taylor, K., & Rohrer, D. (2010). The effects of interleaved practice. *Applied Cognitive Psychology, 24*(6), 837–848. https://doi.org/10.1002/acp.1598

van der Wal, O., Lesci, P., Müller-Eberstein, M., Saphra, N., Schoelkopf, H., Zuidema, W., & Biderman, S. (2025). PolyPythias: Stability and outliers across fifty language model pre-training runs. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2503.09543

Wahlheim, C. N., Dunlosky, J., & Jacoby, L. L. (2011). Spacing enhances the learning of natural concepts: An investigation of mechanisms, metacognition, and aging. *Memory & Cognition, 39*(5), 750–763. https://doi.org/10.3758/s13421-010-0063-y

Warstadt, A., Mueller, A., Choshen, L., Wilcox, E., Zhuang, C., Ciro, J., Mosquera, R., Paranjabe, B., Williams, A., Linzen, T., & Cotterell, R. (2023). Findings of the BabyLM Challenge: Sample-efficient pretraining on developmentally plausible corpora. In *Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning* (pp. 1–6). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.conll-babylm.1

Xie, S. M., Pham, H., Dong, X., Du, N., Liu, H., Lu, Y., Liang, P., Le, Q. V., Ma, T., & Yu, A. W. (2023). DoReMi: Optimizing data mixtures speeds up language model pretraining. In *Advances in Neural Information Processing Systems 36*. https://arxiv.org/abs/2305.10429

Yan, V. X., & Sana, F. (2021a). Does the interleaving effect extend to unrelated concepts? *Journal of Educational Psychology, 113*(1), 125–137. https://doi.org/10.1037/edu0000470

Yan, V. X., & Sana, F. (2021b). The robustness of the interleaving benefit. *Journal of Applied Research in Memory and Cognition, 10*(4), 589–602. https://doi.org/10.1016/j.jarmac.2021.05.002

Yang, Y., Jones, M., Mozer, M. C., & Ren, M. (2024). Reawakening knowledge: Anticipatory recovery from catastrophic interference via structured training. In *Advances in Neural Information Processing Systems 37*. https://doi.org/10.52202/079017-2621

Yang, Y., Sun, H., Li, J., Liu, R., Li, Y., Liu, Y., Gao, Y., & Huang, H. (2023). *MindLLM: Pre-training lightweight large language model from scratch, evaluations and domain applications* (arXiv:2310.15777). arXiv. https://arxiv.org/abs/2310.15777

Ye, J., Liu, P., Sun, T., Zhan, J., Zhou, Y., & Qiu, X. (2025). Data mixing laws: Optimizing data mixtures by predicting language modeling performance. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2403.16952

Zheng, J., Cai, X., Qiu, S., & Ma, Q. (2025). Spurious forgetting in continual learning of language models. In *International Conference on Learning Representations (ICLR 2025)*. https://arxiv.org/abs/2501.13453

Zulkiply, N., & Burt, J. S. (2013). The exemplar interleaving effect in inductive learning: Moderation by the difficulty of category discriminations. *Memory & Cognition, 41*(1), 16–27. https://doi.org/10.3758/s13421-012-0238-9
