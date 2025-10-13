# copy-paste-detection

A tool to detect operations of copy-pasting regions of an image onto itself. At its core, we use the PatchMatch algorithm.

## Logic of PatchMatch

1. **Initialization**: Random assignment of matches.

2. **Propagation**: We get the offset for the pixels at the top and at the left translated one pixel to the right or to the bottom if the error is less than the current one at the pixel we are examinating. We then propagate backwards (bottom and right).

3. **Random search**: We get the current offset and search randomly in a ever-decreasing radius to find out if we can improve even more our match.
