<section class="row"
         data-background="images/cover.jpg">

  <div align="left"
       class="col s12">
    <h3 class="mywhite">
      AI Powered
    </h3>
    <h1 class="mywhite">
      Mouse
      <img src="images/mouse.png" width="50px"/>
      Behavior Analyzer
    </h1>
  </div>
  <div class="col
              s12
              mywhite">
    <p>
      [Feng Xia](mailto:noreply@feng.com) | 10/13/2021
    </p>
  </div>
</section>

---

# Table of Contents


---

# background

---

# challenge

why are we doing this

1. Mouse behaviors lack definition and universal standard. Some
   behaviors can be determined as a gesture of a moment, whereas
   others are only meaningful w/ consideration of a sequence of time.
2. Identifying behaviors in a large video set by human operator
   is slow and lack of consistency.
3. Human operator is often over-sensitive to variations in video
   quality such as resolution, lighting, angle. This leads to
   distraction of the research's primary objective, rendering the
   analysis process unreliable.
4. Cycle of analysis iteration from video capture to result analysis
   is long, labor intensive and difficult to replicate, hindering
   research progress.
5. Solutions such as [Simba][https://github.com/sgoldenlab/simba] and
   [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) are popular,
   but unstable, difficult to install, and have a high learning curve
   for new user.

---

# mission statement

Objective and quantitative behavior classification, particularly in a
naturalist environment, would greatly facilitate scientific
research. Leveraging artificial intelligence tools, our goal is to
provide a technical platform for efficient and rigorous behavioral
classification in a user-friendly interface.

We use computer's image processing power to identify mouse behavior
through **image pattern recognition**.  By leveraging **artificial
intelligence (AI)** and well-known **deep learning models**, we build
an application to detect a behavior defined by the user in a large
number of video **efficiently** and **consistently**. With a **lower
learning curve** than alternative solutions, this framework gives
mouse behavior researchers and teams access to the fruit of AI and
computer science that will empower their subject endeavor.

---

# end-to-end workflow

An end-to-end workflow defines the overall infrastructure of the
proposed solution:

1. **video processing**: create video &rarr; editing &rarr; tagging
frames &rarr; categorization

2. **training**: training set &rarr; AI engine &rarr; _what is good enough?_

3. **testing**: `A` is a behavior; `!A` is any behavior other than `A`

  - positive: A &rarr; A
  - false negative: A &rarr; !A
  - negative: !A &rarr; !A
  - false positive: !A &rarr; A

4. **batching**: untagged set &rarr; analyzer &rarr; behavior decisions

5. **reporting**: behavior decisions &rarr; analysis data set including
   interim data, charts, and reports.

6. **publishing**: analysis data set &rarr; public facing web site &
   other venues

---

# system design

![](images/deployment%20arch.png)

---

# project planning

- [planning](./downloads/Overview.html)
- [project status report](/.downloads/Status.html)
- [resource allocation report](./downloads/Development.html)
- [resource workload report](./downloads/ResourceGraph.html)
- [contact list](./downloads/ContactList.html)

---

Enjoy ~~
# [&rArr;](https://fengxia41103.github.io/stock/dev%20and%20deployment.html)
