# CMU ASF/AMC adapter

Use `--source cmu_asf`. Each subject directory must contain its matching ASF;
an ancestor must contain `cmu_metadata.json`, cached by
`tools/booster_k1_retarget/catalog_cmu.py` from official subject pages.
Unknown per-trial FPS is rejected, never silently set to 120 Hz.

The parser uses column-vector ASF basis conjugation, full hierarchy FK,
the CMU meter scale `0.0254 / ASF length`, and anatomical rest-direction
alignment before approximate SMPL rotation transfer. Unsupported translation/
length channels, nonstandard channel orders, and nonzero root calibration
fail explicitly. Shape is neutral, not fitted. Validate source skeleton and
canonical/K1 renders before bulk production; matching array shapes is not
proof of correct motion semantics.

References:
- http://mocap.cs.cmu.edu/faqs.php
- https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html
