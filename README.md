java c
Module code and Title 
DTS304TC Machine Learning School Title 
School of AI and Advanced Computing Assignment Title 
Assessment Task 2 
DTS304TC Machine Learning 
Coursework – Assessment Task 2
Submission deadline: TBD
Percentage in final mark: 50%
Learning outcomes assessed: C, D
●             Learning   outcome   C:   apply   ML   algorithms   for   specific problems.
●             Learning   outcome   D:   Demonstrate   proficiency   in   identifying   and   customizing   aspects   on   ML   algorithms   to   meet particular   needs.
Individual/Group: Individual
Length: This   assessment   comprises   a   guided   coding   and   experimentation project,   along   with   an   associated research report.
Late   policy: 5%   of   the   total   marks   available   for the   assessment   shall be   deducted   from   the   assessment
mark   for   each   working   day   after   the   submission   date,   up   to   a   maximum   of   five   working   days
Risks: 
•      Please   read   the   coursework   instructions   and   requirements   carefully. Not   following   these   instructions   and   requirements   may   result   in   a   loss   of   marks.
•      The   formal   procedure   for   submitting   coursework   at   XJTLU   is   strictly   followed.   Submission   link   on   Learning   Mall   will   be provided   in   due   course.   The   submission   timestamp   on   Learning   Mall   will be      used   to   check   late   submission.
Guided Project: Street-view Semantic Segmentation with U-Net 
In this coursework,   you will explore street-view      semantic segmentation using a   U-Net   model across   multiple   challenging   scenarios.
. Part   A focuses   on   evaluating   a   pretrained   daytime   segmentation   model.   You   will   write   your   own   evaluation script.    and   compute   metrics such as   Dice and accuracy. Then,   by   examining   both   successful    and    failed      segmentations,   you   will   highlight   the   difficulties of segmenting      diverse   daytime   scenes   and   discuss why   semantic   segmentation   can   be   challenging.
. Part   B shifts   attention to   night-time   images.   You   will   evaluate   the   same   pretrained   model   on   night-   time data, then implement    fine-tuning    using   various   loss   functions   (e.g.,   Dice,   Cross-Entropy,   combinations of both, or other   losses   of interest).   In addition, you   will experiment   with one   additional    training    technique of your choice—such as different data augmentations, different   network   learning-rate   schedulers,   optimizers   (including   momentum   training),   or   layer-wise   fine-   tuning—and   compare the results,   explaining the   outcomes. You will   also discuss   from   a theoretical standpoint   why   combining   Dice   and   Cross-Entropy   losses   can   be   advantageous   and   illustrate   how   fine-tuning   improves performance   on   night-time   images.
. Part C involves retraining   a U-Net   from   scratch   on   daytime   images, with   a   focus   on   small   or   distant   objects   (e.g.,   people   or   cars   far   away   in   the   scene).   You   are   encouraged   to   propose   and   implement   one or two strategies—such as incorporating object detection, targeted data augmentation,   different   loss    functions,    or   multi-scale architectures—to enhance segmentation performance    for   these      harder-to-segment classes. You will provide both quantitative metrics and qualitative   analyses   of   your results.
. Part    D offers   an   opportunity   to   reflect   on   the   interesting   and   novel   aspects   of    the    work    you   developed   during   the   coursework,   the   principal   challenges   you   faced,   how   you   addressed   them,   and the lessons you    learned.   Your   coding   quality will   also   be   evaluated, so   focus   on   clarity,   organization, and efficiency. Completing this coursework will deepen your understanding of   semantic segmentation    metrics,    fine-tuning    techniques,    network   modifications   for   small   object   detection,   and the practical   challenges   of   deploying   segmentation models under varied   lighting   and   scene   conditions.

The   assessment   comprises   two parts:
1. Coding and experimentation 
2. Completing the assessment work sheet 
As   you   work   on   the   coding   and   experimentation,   it   is   crucial   to   record   and   document   all   relevant   findings;   these   will   be   integrated   into   your   assessment   sheet.
Before   starting   this project, please   ensure   you   have   downloaded   the   pretrained   model,   the   assessment   codebase,   and   the   example   Jupyter   notebook.
Notes: 
1.          A   sufficiently powerful   GPU   is   required to   complete   this   coursework.   You   may   use   the   GPU
machines   available   in   our   school   labs   (e.g.,   FYP   labs).   Please   schedule   and   coordinate usage   with   your   classmates   well   in   advance   to   avoid   conflicts.
2.          You   are   strongly   encouraged   to   review   the provided   code   and training   notebook,   as   you   willrefer   to   and   modify   them   throughout   the   project.   Feel   free   to   use   code   covered   during   our   lab   sessions   or   other   online   resources.   However,   please   provide   proper   citations   and   links   to   any      external   resources   you use.
Important: You may   not use   ChatGPT   to   directly   generate   answers   for the   coursework.   It   is important   to   think   critically   and   develop   your   own   solutions—by yourself or   through   reputable   online   resources—to   strengthen   your research   and   critical   thinking   skills.   You may,   however, use   ChatGPT   for   code   understanding   or   debugging,   as well   as   for   grammar   and   writing assistance.   If   you   do   so,   use   it responsibly,   ensuring   you   fully   understand   and   take   ownership   of   all   the   content   in   your   coursework.(A) Evaluate the Pretrained Model on Bright Sunny Test   Images   (10   marks) Complete   the   supplied   notebook   dts304_a2_part1.ipynb.   The   pretrained model   and   testing   Dataloader code   are   provided,   and   you   are   required   to   implement   both   the   testing   loop   and   the   evaluation   metrics.
(A1) Testing Loop 
.               Implement   the   testing   loop to   generate   predictions   for   the   test   images using   the   pretrained   model.
.               Ensure   the   model   is   in   evaluation   mode   (model.eval())   and   use   torch.no_grad()   to   optimize   performance   and   accuracy.
(A2) Accuracy Metrics 
1.             Within the testing loop, calculate   global   image   accuracy   and   per-class   accuracy   for   each image,   as   well   as   for the   overall   dataset.  
.            Global   Image   Accuracy
.             Defined   as   the   percentage   of correctly   classified pixels   per   image.
.             The   overall   dataset   global image   accuracy   is the   average of   the   global accuracies   across   all   images.  
.            Per-Class   Accuracy   (IOU)
.             Use Intersection-over-Union (IOU)   for each   of   the    14 categories.
.             Compute   mean-IOU by   averaging   across the    14   categories.
.             If a   category   does   not   appear   in   the   ground   truth   or   prediction,   assign   NaN   and   use   np. nanmean   for   the   mean-IOU   calculation.
.             The   overall   dataset per-class IOU   is   obtained   by   averaging   the   per-class   IOU   scores   for   each   category   across   all   images.
2.             Calculate the   overall   dataset global image accuracy   and the   overall   dataset   per-class   IOU scores. Note   any   significant   variations   in   the   IOU   scores   across   different   classes   and   discuss   potential   reasons   for   these   differences. 
(A3)   Rank the   images based   on   accuracy   and   IoU   scores,   then   select   a   few   examples   where   the   model   performs   exceptionally   well   and   others   where   it   struggles.   Use   these   selections   to   illustrate   the strengths   and   limitations   of   your   street-view   segmentation   approach.   For   example,   the   model   might   accurately   segment   large,   well-lit   vehicles   but   struggle   with   small   or   distant   pedestrians,   poorly illuminated regions,   or   overlapping   objects.   In   each   case,   explain   why   these   challenges   arise   (e.g., limited pixel   information,   lighting   variations,   or   complex   backgrounds),   and use your   chosen   images   to   demonstrate   these   issues.
B) Fine-Tune the Pretrained Model on the Cloudy Dataset   (30   marks)Finish   the   supplied   notebook   dts304_a2_part2.ipynb.   The   pretrained model   and   testing   data   loader   code      for the   cloudy   dataset   are   provided.   You   will   evaluate   the   model’s   current performance,   implement   fine-   tuning   using   various   loss   functions,   and   then   re-evaluate   and   analyse   the   results.
(B1) Evaluate   Cloudy Dataset   Performance. Use the   code   from Part   1   (or   a   similar   approach)   to
evaluate   the   model’s performance   on   the   cloudy   dataset   by   calculating   Dice   and   accuracy   (or   other
relevant metrics).   Since   cloudy   conditions   typically   present   additional   challenges, you   may   observe   a
decrease   in   segmentation   quality   compared to   t代 写DTS304TC Machine Learning Assessment Task 2R
代做程序编程语言he   sunny   scenes   tested   in   Part    1.   This   lower performance   highlights   the   need   for   further   enhancements—such   as model   finetuning—to   improve   the   model’s
robustness   under varying   weather   conditions.
(B2) Fine-Tuning 
1.          Adapt   Training Notebook:   Modify the   notebook to   fine-tune   the   model   on   the   cloudy dataset.
2.          Loss Functions:   Experiment with   different   loss   functions   (e.g.,   Dice,   CE,   Dice   + CE)
during   the   finetuning   and   analyse   their   impact   on   performance.
3.          Additional   Training   Technique:   Implement   at   least   one   additional technique   of   your
choice   (e.g.,   data   augmentations,   alternative   optimizers   with   momentum,   learning-rate   scheduling,   or   selective   layer   fine-tuning).
4.          Training Process Monitoring.   Plot   and   compare training   vs.   validation   loss to   track
improvements.   Include   at   least   one   plot   in   your   assessment   worksheet   to   demonstrate   how   you   selected   the   best   fine-tuned model.
(B3) Results and   Analysis 
1.          Loss Function   Insights:   Discuss the   theoretical/algorithm   properties   of   Dice   and   Cross-Entropy   (CE)   losses,   including   why their   combination may   offer unique   advantages   (e.g.,   gradient   considerations).   Compare   these   theoretical   insights   to   your   empirical   findings by   reporting   accuracy   and   IOU   scores   for   each   loss   configuration—Dice-only,   CE-only,   and Dice   +   CE.
2.          Enhancements    Experiments:   Summarize   the   additional technique(s) you   employed—
such   as   new   augmentations,   different   optimizers,   or   layer-wise   training—by   presenting   both   theoretical/algorithmic rationale   and   the   observed   experimental   outcomes.
3.          Re-Evaluation:   After   selecting the most   effective   loss   function   and training
enhancements,   re-test   your best   fine-tuned   model   using   the   same   procedure   as   the   initial   cloudy   dataset   evaluation.   Present   global   image   accuracy   and per-class   IOU   results   and      compare   them   against   the   baseline   to   highlight   any   performance   gains.
C) Retrain From Scratch on daytime images  Enhance   Small-Object   Segmentation (35 Marks) 
You   are provided with   the notebook   dts304_a2_part3.ipynb, which   you   will   use   to   train   a   segmentation   model   on   daytime   images. Your primary   objective   is to   improve   segmentation   accuracy   for   small objects—specifically   cars, pedestrians,   and lane-markings—while maintaining   or   improving   overall   performance   on   other   classes.
(C1) Small-Object Challenge Evaluation (Cars, Pedestrians, Lane-Markings) 
1.          Why   small   objects   are   difficult:   Distant   cars   and pedestrians   often   occupy   only   a   few pixels,
leading   to   lower   accuracy   or   missed   detections.   Lane   markings   can   be   thin   and   fragmented,   causing   false   negatives   and reduced   IoU.
2.          You   are   given   a   function   called   compute_object_level_stats_percentile   that:
.          Splits   objects   into   “small”   or   “large”   based   on   a   threshold   (e.g.,   number   of   pixels)   for   each   category.
.          Computes   precision,   recall,   and   F1-score   for   each   category   (small   vs.   large   objects).   Use   this   function   to   analyse how   well the   model   segments   small vs.   large   cars, pedestrians,   and   lane   markings.
3.         Tasks:
.          Calculate   Small/Large   Object Metrics:   Run   compute_object_level_stats_percentile   for   car, lane-mark,   and pedestrian classes.
.          Compare Precision, Recall, F1:   Compare small-object   metrics with   large-object   metrics to   validate   that   segmentation   of small   objects   is   indeed   a   challenge   for   the         car,   lane-marker,   and pedestrian   classes.
.          Provide Visual   Examples:   Show   at   least   five   images with   ground-truth
segmentations   vs.   model   prediction   results   highlighting   missed   or partially   segmented   small   objects.
(C2) Strategies to Improve Small-Object Segmentation 
After   identifying   weaknesses   in   small-object   segmentation,   choose   at   least   one   of   the   following   strategies   (implementing two strategies may yield higher marks):
1.         Combine Semantic Segmentation with Object Detection
.          Use   an   object   detector   to   first   locate   small   objects   (cars/pedestrians),   then   apply   a   segmentation   network   to   those   regions.
.             Or   run   a   segmentation   model   and   a   detection   model   in parallel,   fusing   their   outputs   for   improved   small-object   accuracy.
.          You   may   use   a   pre-trained   object   detection   model   downloaded   from   the   internet   to   detect   cars   and pedestrians.   If   you   choose   this   method,   it   is   acceptable   to   omit
experiments   for   the   'lane-marker'   class   if   no   suitable   object   detection   model   is   available   for   it.
2.         Custom Data Augmentation
.          Example:   Randomly   zoom   in   on   regions   containing   small   objects,   increasing   their   resolution   during   training.
.          Alternatively,   duplicate   or   scale   small   objects   synthetically   to   increase   their   representation   in   the   training   set.
3.         Pixel Weighting Map
.          Assign   higher   loss   weights   to   pixels   belonging   to   small   objects
.          Ensures   the   model   pays   more   attention   to   the   pixels   of small   objects   but   requires   careful   tuning   to   avoid   overfitting.
4.         Multi-Scale / Pyramid Architectures
.          Incorporate   modules   like   ASPP   (Atrous   Spatial   Pyramid   Pooling)   to   capture   features   at   multiple   scales.
.          Helps   detect   smaller   objects   without   losing   overall   context.
5.         Other approaches
.          You   can   select   other   approaches   as   you   wish   (for   example,   different   loss   function).   .          However,   if   your   approach   is   overly   simplistic   or   only   involves   straightforward
parameter   tuning   (e.g.,   adjusting   image   resolution,   learning   rate,   or   network   parameters),   it   will   not   be   awarded   high   marks.Modify   or   create   a new   training   loop   in   dts304_a2_part3.ipynb   to   include   your   chosen   technique(s).      Clearly   mark   or   comment   your   changes.   Provide   all   source   code   in   your   submission   files   and briefly   mention   modifications   in your   assessment   worksheet.
(C3) Results  Analysis 
1.         Post-Training Evaluation
.          Recompute   overall   metrics   (global   accuracy,   mean   IoU)   as   you   did   for   the   original   model.
.          Use   compute_object_level_stats_percentile   again   to   see   how   your   improvements
affect   small-object metrics   for   cars,   pedestrians,   and   lane-markings.
.          Compare   (precision_small,   recall_small,   F1_small)   before   and   after   your   changes.
2.         Highlight Gains
.          Discuss   what   strategies   (1   or   2   strategies)   have   you   made   to   improve   small   object   segmentation   and   explain   their   algorithm   principles.
.          If   you   employed   multiple   improvements   or   strategies,   clearly   explain   how   each
individual   approach/strategy   contributed   to   the   enhancement   of small-object
segmentation   based   on   experimental   evaluation   results   (for   example,   the   small-object   metrics, mean   IOU   or   global   accuracy)   and/or   your   analysis.
.          Note   any   trade-offs,   such   as   slightly   lower   performance   on   large   objects   or   other   classes.
3.         Qualitative Examples
.             Show   side-by-side   images   (ground   truth   vs.   predicted   segmentation)   demonstrating   improvements   in   small-object   detection.
.          Point   out   any   remaining   challenges   (extremely   tiny   objects,   occlusions,   etc.).
D) Coursework Reflections (15 Marks) 
.             Write   a   brief   reflection   highlighting   unique   or   interesting   algorithms   or   strategies   you   implemented   in   the   coursework.
.             Reflect   on   the   challenges   you   personally   faced   during   the   coursework,   describing   the   efforts   you   made   to   overcome   them   and   the   key   lessons   you   learned.
.             Based   on   your   reflections,   discuss   potential   future   work   that   could   further   enhance   semantic   segmentation   in   street-view   applications.
E) Coding Quality, Worksheet Quality and Adherence to Submission   Guidelines (10 Marks) 
.             Ensure   your   code   is   clear,   easy   to   understand,   and   includes   well-written   comments   to   explain   its   functionality.
.             Follow   the   worksheet   template,   formatting   guidelines,   and   all   prescribed   requirements   for   the   submission.
.             Write   in   clear   and   proper   English,   keeping   your   answers   concise   and   within   the   specified word   limit.   Pay   close   attention   to   maintaining   excellent   formatting   throughout.





         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
