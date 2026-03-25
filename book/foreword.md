# Foreword

**Author:** Brock Webb

*The views expressed in this work are those of the author in his personal capacity and do not represent the views of any federal agency. This work was not produced in the author's official capacity and does not constitute an official publication of any federal agency.*

---

## On the Speed of This Field

This is a fast-moving field. Parts of this book will be outdated by the time you read them. The specific tools, models, and APIs mentioned here will evolve. What will not change are the principles: understand your methods, validate your results, document your decisions, and maintain human judgment over automated systems.

Read for the principles. Treat the specific tools as illustrations.

---

## Classical ML and Generative AI: Choose the Right Tool

This book covers both traditional machine learning methods and the newer class of generative AI tools. They are not interchangeable.

Classical methods (regression, random forests, clustering) are well-understood, deterministic, and explainable. When they solve your problem, use them. Do not reach for a generative AI tool where a perfectly good classical method will do. Don't chase shiny.

Generative AI tools incur what I call a *stochastic tax*: their outputs are variable, sometimes unpredictably so. As statisticians, we are comfortable with variability. We understand variance, we measure it, we try to reduce it. Statistical quality control, acceptable defect rates, tolerance intervals: these are our tools. Nothing, including human judgment, is infallible. Humans are just as fallible as our AI partners. The real world is messy, nonlinear, nondeterministic, and always full of surprises. We plan for the worst, do our best, and continuously improve as we gain new knowledge. That is science.

The right approach is to match the problem with the right tool. Sometimes that means generative AI. Often it means something simpler.

---

## What AI Changes (and What It Doesn't)

AI does not change what we have been doing so much as it makes visible areas we may have been underinvesting in: documentation, governance, traceability, provenance of data and results, and evidence-based validation of pipeline integrity.

These are not new requirements. They are requirements that become acute when your research instrument has mutable internal state and can confabulate its own history. Knowing who is accountable and what policies apply before you touch AI, and understanding what could go wrong with your specific use case: these are not bureaucratic overhead. They are the foundation on which trustworthy AI-assisted work rests. Chapter 14 introduces the formal frameworks.

Chapter 15 introduces a formal framework for this problem: State Fidelity Validity. The question it answers is whether the AI pipeline you built last Tuesday still reflects what you decided last Tuesday, or whether something drifted between then and now.

---

## On AI-Assisted Authoring

This book was written with the assistance of AI tools. The pedagogical decisions, domain adaptation, audience framing, original chapters, and all editorial judgments are mine. The AI did execution work: drafting, reformatting, code generation. I did the authoring: deciding what to teach, how to frame it, what to cut, what to write from scratch, and what to validate.

This is consistent with the book's central argument: AI assists, humans decide.

---

## Acknowledgments

The idea that domain experts need their own AI curriculum was inspired by Zhiling Zheng's AI4CHEM course at Washington University in St. Louis.

*Throughout my career, I have been continually humbled and inspired by the generosity of others: those who shared their knowledge, offered honest feedback, and walked with me for portions of this journey. There are too many to name individually, and attempting to do so would inevitably leave someone out. You know who you are, and I am grateful. The only way I know to truly show that gratitude is to give back. This work is one small attempt to do so.*
