/************************************************************************
  Copyright (c) 2003. David G. Lowe, University of British Columbia.
  This software is being made available for research purposes only
  (see file LICENSE for conditions).  This notice must be retained on
  all copies or modifications of this software.
 *************************************************************************/

/* key.cpp:
   This file contains code to extract invariant keypoints from an image.
   The main routine is GetKeypoints(image) which returns a list of all
   invariant features from an image.
 */

#include "Sift.h"

using namespace Lowe;

//------------------------------------------------------------------------------
// Constants from the former util.c
//------------------------------------------------------------------------------
const double SIFT::pi = 3.1415927;

const int SIFT::OriSize = 8;
const int SIFT::IndexSize = 4;
const int SIFT::VecLength = IndexSize * IndexSize * OriSize;

const int SIFT::MagFactor = 3;

const float SIFT::GaussTruncate = 4.0;

//------------------------------------------------------------------------------
// Constants from the former key.c
//------------------------------------------------------------------------------
const bool SIFT::DoubleImSize = false;

const int SIFT::Scales = 3;

const float SIFT::InitSigma = 1.6;

const int SIFT::BorderDist = 5;

float SIFT::PeakThreshInit = 0.01;

const float SIFT::EdgeEigenRatio = 10.0;//2.5;

const bool SIFT::UseHistogramOri = true;

const int SIFT::OriBins = 36;

const float SIFT::OriSigma = (UseHistogramOri ? 1.5 : 1.0);

const float SIFT::OriHistThresh = 0.8;

const float SIFT::IndexSigma = 1.0;

const bool SIFT::IgnoreGradSign = false;

const float SIFT::MaxIndexVal = 1.0;//0.2;

const bool SIFT::SkipInterp = false;

//------------------------------------------------------------------------------
SIFT::SIFT() {
    m_bUninitializedH = true;
}

//------------------------------------------------------------------------------
// Methods from the former key.c
//------------------------------------------------------------------------------
SIFT::KeyList SIFT::getKeypoints(const char* szImageFile) {
    FILE* pf = fopen(szImageFile, "rb");
    if (!pf) {
        std::cout << "Cannot open " << szImageFile << std::endl;
        return std::vector<Key>();
    }
    Image image = ReadPGM(pf);
    KeyList keys = GetKeypoints(image);

    fclose(pf);
    return keys;
}

//------------------------------------------------------------------------------
SIFT::KeyList SIFT::getKeypoints(const Image& image) {
    return GetKeypoints(image);
}

//------------------------------------------------------------------------------
SIFT::KeyList SIFT::getKeypoints(const char* szImageFile, Image& rImage) {
    FILE* pf = fopen(szImageFile, "rb");
    if (!pf) {
        std::cout << "Cannot open " << szImageFile << std::endl;
        return std::vector<Key>();
    }
    rImage = ReadPGM(pf);
    KeyList keys = GetKeypoints(rImage);

    fclose(pf);
    return keys;
}

//------------------------------------------------------------------------------
void SIFT::writeKeypoints(const char* szOutputFile, const KeyList& keys) {
    FILE* pf = fopen(szOutputFile, "wt");
    if (!pf) {
        std::cout << "Cannot open " << szOutputFile << std::endl;
        return;
    }
    WriteKeypoints(pf, keys);
    fclose(pf);
}

//------------------------------------------------------------------------------
void SIFT::drawKeypoints(Image& im, const KeyList& keys)
{
    int count = 0;
    KeyList::const_iterator k;

    for (k = keys.begin(); k < keys.end(); ++k) {

        /* Draw 3 lines creating a horizontal arrow of unit length.
           Each line will be transformed according to keypoint parameters.
         */
        TransformLine(im, *k, 0.0, 0.0, 1.0, 0.0);  /* Main shaft of arrow. */
        TransformLine(im, *k, 0.85, 0.1, 1.0, 0.0);
        TransformLine(im, *k, 0.85, -0.1, 1.0, 0.0);
        count++;
    }
    fprintf(stderr, "%d keypoints found.\n", count);
}

//------------------------------------------------------------------------------
SIFT::KeyList SIFT::GetKeypoints(const Image& inputImage)
{
    int minsize;
    float curSigma, sigma, octSize = 1.0;
    KeyList keys;   /* List of keypoints found so far. */
    keys.reserve(1000);   // preallocate the vector
    Image image, nextImage;

    PeakThresh = PeakThreshInit / Scales;

    /* If DoubleImSize flag is set, then double the image size prior to
       finding keypoints.  The size of pixels for first octave relative
       to input image, octSize, are divided by 2.
     */
    if (DoubleImSize) {
        image = DoubleSize(inputImage);
        octSize *= 0.5;
    }
    else {
        image = inputImage;
    }

    /* Apply initial smoothing to input image to raise its smoothing
       to InitSigma.  We assume image from camera has smoothing of
       sigma = 0.5, which becomes sigma = 1.0 if image has been doubled.
     */
    curSigma = (DoubleImSize ? 1.0 : 0.5);
    if (InitSigma > curSigma) {
        sigma = sqrt(InitSigma * InitSigma - curSigma * curSigma);
        GaussianBlur(image, sigma);
    }

    /* Examine one octave of scale space at a time.  Keep reducing
       image by factors of 2 until one dimension is smaller than
       minimum size at which a feature could be detected.
     */
    minsize = 2 * BorderDist + 2;
    while (image.rows > minsize &&  image.cols > minsize) {
        OctaveKeypoints(image, nextImage, octSize, keys);
        image = HalfImageSize(nextImage);
        octSize *= 2.0;
    }
    /* Release all memory used to process this image. */
    return keys;
}

//------------------------------------------------------------------------------
void SIFT::OctaveKeypoints(const Image& image, Image& pnextImage,
        float octSize, KeyList& keys)
{
    int i;
    float sigmaRatio, prevSigma, increase;
    Image blur[Scales+3], dogs[Scales+2];

    /* Ratio of each scale to the previous one.  The parameter Scales
       determines how many scales we divide the octave into, so
       sigmaRatio ** Scales = 2.0.
     */
    sigmaRatio = pow(2.0, 1.0 / (float) Scales);

    /* Build array "blur", holding Scales+3 blurred versions of the image. */
    blur[0] = image;          /* First level is input to this routine. */
    prevSigma = InitSigma;    /* Input image has InitSigma smoothing. */

    /* Form each level by adding incremental blur from previous level.
       Increase in blur is from prevSigma to prevSigma * sigmaRatio, so
       increase**2 + prevSigma**2 = (prevSigma * sigmaRatio)**2
     */
    for (i = 1; i < Scales + 3; i++) {
        blur[i] = blur[i-1];
        increase = prevSigma * sqrt(sigmaRatio * sigmaRatio - 1.0);
        GaussianBlur(blur[i], increase);
        prevSigma *= sigmaRatio;
    }

    /* Compute an array, dogs, of difference-of-Gaussian images by
       subtracting each image from its next blurred version.
     */
    for (i = 0; i < Scales + 2; i++) {
        dogs[i] = blur[i];
        SubtractImage(dogs[i], blur[i+1]);
    }

    /* Image blur[Scales] has twice the blur of starting image for
       this octave, so it is returned to downsample for next octave.
     */
    pnextImage = blur[Scales];

    /*return */FindMaxMin(dogs, blur, octSize, keys);
}

//------------------------------------------------------------------------------
void SIFT::FindMaxMin(const Image *dogs, const Image *blur,
        float octSize, KeyList& keys)
{
    int s, r, c, rows, cols;
    float val;
    Image map, grad, ori;

    rows = dogs[0].rows;
    cols = dogs[0].cols;

    /* Create an image map in which locations that have a keypoint are
       marked with value 1.0, to prevent two keypoints being located at
       same position.  This may seem an inefficient data structure, but
       does not add significant overhead.
     */
    map.setSize(rows, cols);
    for (r = 0; r < rows; r++)
        for (c = 0; c < cols; c++)
            map.pixels[r][c] = 0.0;

    /* Search through each scale, leaving 1 scale below and 1 above.
       There are Scales+2 dog images.
     */
    for (s = 1; s < Scales+1; s++) {

        /* For each intermediate image, compute gradient and orientation
           images to be used for keypoint description.
         */
        grad.setSize(rows, cols);
        ori.setSize(rows, cols);
        GradOriImages(blur[s], grad, ori);

        const FloatMatrix& pix = dogs[s].pixels;   /* Pointer to pixels for this scale. */

        /* Only find peaks at least BorderDist samples from image border, as
           peaks centered close to the border will lack stability.
         */
        assert(BorderDist >= 2);
        for (r = BorderDist; r < rows - BorderDist; r++)
            for (c = BorderDist; c < cols - BorderDist; c++) {
                val = pix[r][c];       /* Pixel value at (r,c) position. */

                /* DOG magnitude must be above 0.8 * PeakThresh threshold
                   (precise threshold check will be done once peak
                   interpolation is performed).  Then check whether this
                   point is a peak in 3x3 region at each level, and is not
                   on an elongated edge.
                 */
                if (fabs(val) > 0.8 * PeakThresh  &&
                        LocalMaxMin(val, dogs[s], r, c) &&
                        LocalMaxMin(val, dogs[s-1], r, c) &&
                        LocalMaxMin(val, dogs[s+1], r, c) &&
                        NotOnEdge(dogs[s], r, c))
                    InterpKeyPoint(dogs, s, r, c, grad, ori, map, octSize,
                            keys, 5);
            }
    }
    //return keys;
}

//------------------------------------------------------------------------------
bool SIFT::LocalMaxMin(float val, const Image& dog, int row, int col)
{
    int r, c;
    const FloatMatrix& pix = dog.pixels;

    /* For efficiency, use separate cases for maxima or minima, and
       return as soon as possible. */
    if (val > 0.0) {
        for (r = row - 1; r <= row + 1; r++)
            for (c = col - 1; c <= col + 1; c++)
                if (pix[r][c] > val)
                    return false;
    } else {
        for (r = row - 1; r <= row + 1; r++)
            for (c = col - 1; c <= col + 1; c++)
                if (pix[r][c] < val)
                    return false;
    }
    return true;
}

//------------------------------------------------------------------------------
int SIFT::NotOnEdge(const Image& dog, int r, int c)
{
    float H00, H11, H01, det, trace, inc;
    const FloatMatrix& d = dog.pixels;

    /* Compute 2x2 Hessian values from pixel differences. */
    H00 = d[r-1][c] - 2.0 * d[r][c] + d[r+1][c];
    H11 = d[r][c-1] - 2.0 * d[r][c] + d[r][c+1];
    H01 = ((d[r+1][c+1] - d[r+1][c-1]) - (d[r-1][c+1] - d[r-1][c-1])) / 4.0;

    /* Compute determinant and trace of the Hessian. */
    det = H00 * H11 - H01 * H01;
    trace = H00 + H11;

    /* To detect an edge response, we require the ratio of smallest
       to largest principle curvatures of the DOG function
       (eigenvalues of the Hessian) to be below a threshold.  For
       efficiency, we use Harris' idea of requiring the determinant to
       be above a threshold times the squared trace.
     */
    inc = EdgeEigenRatio + 1.0;
    return (det * inc * inc  > EdgeEigenRatio * trace * trace);
}

//------------------------------------------------------------------------------
void SIFT::InterpKeyPoint(const Image* dogs, int s, int r, int c,
        const Image& grad, const Image& ori, Image& map,
        float octSize, KeyList& keys, int movesRemain)
{
    int newr = r, newc = c;
    float offset[3], octScale, peakval;

    /* The SkipInterp flag means that no interpolation will be performed
       and the peak will simply be assigned to the given integer sampling
       locations.
     */
    if (SkipInterp) {
        assert(UseHistogramOri);    /* Only needs testing for this case. */
        if (fabs(dogs[s].pixels[r][c]) >= PeakThresh) {
            AssignOriHist(grad, ori, octSize,
                    InitSigma * pow(2.0, s / (float) Scales),
                    (float) r, (float) c, fabs ( dogs[s].pixels[r][c] ),keys);
        }
        return;
    }

    /* Fit quadratic to determine offset and peak value. */
    peakval = FitQuadratic(offset, dogs, s, r, c);

    /* Move to an adjacent (row,col) location if quadratic interpolation
       is larger than 0.6 units in some direction (we use 0.6 instead of
       0.5 to avoid jumping back and forth near boundary).  We do not
       perform move to adjacent scales, as it is seldom useful and we
       do not have easy access to adjacent scale structures.  The
       movesRemain counter allows only a fixed number of moves to
       prevent possibility of infinite loops.
     */
    if (offset[1] > 0.6 && r < dogs[0].rows - 3)
        newr++;
    if (offset[1] < -0.6 && r > 3)
        newr--;
    if (offset[2] > 0.6 && c < dogs[0].cols - 3)
        newc++;
    if (offset[2] < -0.6 && c > 3)
        newc--;
    if (movesRemain > 0  &&  (newr != r || newc != c)) {
        InterpKeyPoint(dogs, s, newr, newc, grad, ori, map, octSize,
                keys, movesRemain - 1);
        return;
    }

    /* Do not create a keypoint if interpolation still remains far
       outside expected limits, or if magnitude of peak value is below
       threshold (i.e., contrast is too low).
     */
    if (fabs(offset[0]) > 1.5  || fabs(offset[1]) > 1.5  ||
            fabs(offset[2]) > 1.5 || fabs(peakval) < PeakThresh)
        return;

    /* Check that no keypoint has been created at this location (to avoid
       duplicates).  Otherwise, mark this map location.
     */
    if (map.pixels[r][c] > 0.0)
        return;
    map.pixels[r][c] = 1.0;

    /* The scale relative to this octave is given by octScale.  The scale
       units are in terms of sigma for the smallest of the Gaussians in the
       DOG used to identify that scale.
     */
    octScale = InitSigma * pow(2.0, (s + offset[0]) / (float) Scales);

    if (UseHistogramOri) {
        AssignOriHist(grad, ori, octSize, octScale, r + offset[1],
                      c + offset[2], fabs(peakval), keys);
    }
    else {
        AssignOriAvg(grad, ori, octSize, octScale, r + offset[1],
                     c + offset[2], fabs(peakval), keys);
    }
}

//------------------------------------------------------------------------------
void SIFT::AssignOriHist(const Image& grad, const Image& ori, float octSize,
                         float octScale, float octRow, float octCol, float strength, KeyList& keys)
{
    int i, r, c, row, col, rows, cols, radius, bin, prev, next;
    float hist[OriBins], distsq, gval, weight, angle, sigma, interp,
          maxval = 0.0;

    row = (int) (octRow+0.5);
    col = (int) (octCol+0.5);
    rows = grad.rows;
    cols = grad.cols;

    for (i = 0; i < OriBins; i++)
        hist[i] = 0.0;

    /* Look at pixels within 3 sigma around the point and sum their
       Gaussian weighted gradient magnitudes into the histogram.
     */
    sigma = OriSigma * octScale;
    radius = (int) (sigma * 3.0);
    for (r = row - radius; r <= row + radius; r++)
        for (c = col - radius; c <= col + radius; c++)

            /* Do not use last row or column, which are not valid. */
            if (r >= 0 && c >= 0 && r < rows - 2 && c < cols - 2) {
                gval = grad.pixels[r][c];
                distsq = (r - octRow) * (r - octRow) + (c - octCol) * (c - octCol);

                if (gval > 0.0  &&  distsq < radius * radius + 0.5) {
                    weight = exp(- distsq / (2.0 * sigma * sigma));
                    /* Ori is in range of -pi to pi. */
                    angle = ori.pixels[r][c];
                    bin = (int) (OriBins * (angle + pi + 0.001) / (2.0 * pi));
                    assert(bin >= 0 && bin <= OriBins);
                    bin = LOWE_MIN(bin, OriBins - 1);
                    hist[bin] += weight * gval;
                }
            }
    /* Apply circular smoothing 6 times for accurate Gaussian approximation. */
    for (i = 0; i < 6; i++)
        SmoothHistogram(hist, OriBins);

    /* Find maximum value in histogram. */
    for (i = 0; i < OriBins; i++)
        if (hist[i] > maxval)
            maxval = hist[i];

    /* Look for each local peak in histogram.  If value is within
       OriHistThresh of maximum value, then generate a keypoint.
     */
    for (i = 0; i < OriBins; i++) {
        prev = (i == 0 ? OriBins - 1 : i - 1);
        next = (i == OriBins - 1 ? 0 : i + 1);
        if (hist[i] > hist[prev]  &&  hist[i] > hist[next]  &&
                hist[i] >= OriHistThresh * maxval) {

            /* Use parabolic fit to interpolate peak location from 3 samples.
               Set angle in range -pi to pi.
             */
            interp = InterpPeak(hist[prev], hist[i], hist[next]);
            angle = 2.0 * pi * (i + 0.5 + interp) / OriBins - pi;
            assert(angle >= -pi  &&  angle <= pi);

            /* Create a keypoint with this orientation. */
            MakeKeypoint(grad, ori, octSize, octScale, octRow, octCol,
                    angle, strength, keys);
        }
    }
}

//------------------------------------------------------------------------------
void SIFT::SmoothHistogram(float *hist, int bins)
{
    int i;
    float prev, temp;

    prev = hist[bins - 1];
    for (i = 0; i < bins; i++) {
        temp = hist[i];
        hist[i] = (prev + hist[i] + hist[(i + 1 == bins) ? 0 : i + 1]) / 3.0;
        prev = temp;
    }
}

//------------------------------------------------------------------------------
float SIFT::InterpPeak(float a, float b, float c)
{
    if (b < 0.0) {
        a = -a; b = -b; c = -c;
    }
    assert(b >= a  &&  b >= c);
    return 0.5 * (a - c) / (a - 2.0 * b + c);
}

//------------------------------------------------------------------------------
void SIFT::AssignOriAvg(const Image& grad, const Image& ori, float octSize,
                        float octScale, float octRow, float octCol, float strength, KeyList& keys)
{
    int r, c, irow, icol, rows, cols, radius;
    float gval, sigma, distsq, weight, angle, xvec = 0.0, yvec = 0.0;

    rows = grad.rows;
    cols = grad.cols;
    irow = (int) (octRow+0.5);
    icol = (int) (octCol+0.5);

    /* Look at pixels within 3 sigma around the point and put their
       Gaussian weighted vector values in (xvec, yvec).
     */
    sigma = OriSigma * octScale;
    radius = (int) (3.0 * sigma);
    for (r = irow - radius; r <= irow + radius; r++)
        for (c = icol - radius; c <= icol + radius; c++)
            if (r >= 0 && c >= 0 && r < rows && c < cols) {
                gval = grad.pixels[r][c];
                distsq = (r - octRow) * (r - octRow) + (c - octCol) * (c - octCol);
                if (distsq <= radius * radius) {
                    weight = exp(- distsq / (2.0 * sigma * sigma));
                    /* Angle is in range of -pi to pi. */
                    angle = ori.pixels[r][c];
                    xvec += gval * cos(angle);
                    yvec += gval * sin(angle);
                }
            }
    /* atan2 returns angle in range [-pi,pi]. */
    angle = atan2(yvec, xvec);
    MakeKeypoint(grad, ori, octSize, octScale, octRow, octCol,
            angle, strength, keys);
}

//------------------------------------------------------------------------------
void SIFT::MakeKeypoint(const Image& grad, const Image& ori, float octSize,
        float octScale, float octRow, float octCol, float angle, float strength,
        KeyList& keys)
{
    Key k;
    k.ori = angle;
    k.row = octSize * octRow;
    k.col = octSize * octCol;
    k.scale = octSize * octScale;
    k.strength = strength;
    MakeKeypointSample(k, grad, ori, octScale, octRow, octCol);
    keys.push_back(k);
}

//------------------------------------------------------------------------------
void SIFT::MakeKeypointSample(Key& key, const Image& grad, const Image& ori,
        float scale, float row, float col)
{
    int i, intval;
    bool changed = false;
    float vec[VecLength];

    /* Produce sample vector. */
    KeySampleVec(vec, key, grad, ori, scale, row, col);

    /* Normalize vector.  This should provide illumination invariance
       for planar lambertian surfaces (except for saturation effects).
       Normalization also improves nearest-neighbor metric by
       increasing relative distance for vectors with few features.
     */
    NormalizeVec(vec, VecLength);

    /* Now that normalization has been done, threshold elements of
       index vector to decrease emphasis on large gradient magnitudes.
       Admittedly, the large magnitude values will have affected the
       normalization, and therefore the threshold, so this is of
       limited value.
     */
    for (i = 0; i < VecLength; i++)
        if (vec[i] > MaxIndexVal) {
            vec[i] = MaxIndexVal;
            changed = true;
        }
    if (changed)
        NormalizeVec(vec, VecLength);

    /* Convert float vector to integer. Assume largest value in normalized
       vector is likely to be less than 0.5.
     */
    key.ivec.resize(VecLength);
    for (i = 0; i < VecLength; i++) {
        intval = (int) (512.0 * vec[i]);
        assert(intval >= 0);
        key.ivec[i] = (unsigned char) LOWE_MIN(255, intval);
    }
}

//------------------------------------------------------------------------------
void SIFT::NormalizeVec(float *vec, int len)
{
    int i;
    float val, fac, sqlen = 0.0;

    for (i = 0; i < len; i++) {
        val = vec[i];
        sqlen += val * val;
    }
    fac = 1.0 / sqrt(sqlen);
    for (i = 0; i < len; i++)
        vec[i] *= fac;
}

//------------------------------------------------------------------------------
void SIFT::KeySampleVec(float *vec, const Key& key, const Image& grad,
        const Image& ori, float scale, float row, float col)
{
    int i, j, k, v;
    Float3DArray index(IndexSize);

    /* Initialize index array. */
    for (i = 0; i < IndexSize; i++) {
        index[i].resize(IndexSize);
        for (j = 0; j < IndexSize; j++) {
            index[i][j].resize(OriSize, 0.0f);
        }  // for j
    }  // for i

    KeySample(index, key, grad, ori, scale, row, col);

    /* Unwrap the 3D index values into 1D vec. */
    v = 0;
    for (i = 0; i < IndexSize; i++)
        for (j = 0; j < IndexSize; j++)
            for (k = 0; k < OriSize; k++)
                vec[v++] = index[i][j][k];
}

//------------------------------------------------------------------------------
void SIFT::KeySample(Float3DArray& index,
        const Key& key, const Image& grad, const Image& ori,
        float scale, float row, float col)
{
    int i, j, iradius, irow, icol;
    float spacing, radius, sine, cosine, rpos, cpos, rx, cx;

    irow = (int) (row + 0.5);
    icol = (int) (col + 0.5);
    sine = sin(key.ori);
    cosine = cos(key.ori);

    /* The spacing of index samples in terms of pixels at this scale. */
    spacing = scale * MagFactor;

    /* Radius of index sample region must extend to diagonal corner of
       index patch plus half sample for interpolation.
     */
    radius = 1.414 * spacing * (IndexSize + 1) / 2.0;
    iradius = (int) (radius + 0.5);

    /* Examine all points from the gradient image that could lie within the
       index square.
     */
    for (i = -iradius; i <= iradius; i++)
        for (j = -iradius; j <= iradius; j++) {

            /* Rotate sample offset to make it relative to key orientation.
               Uses (row,col) instead of (x,y) coords.  Also, make subpixel
               correction as later image offset must be an integer.  Divide
               by spacing to put in index units.
             */
            rpos = ((cosine * i + sine * j) - (row - irow)) / spacing;
            cpos = ((- sine * i + cosine * j) - (col - icol)) / spacing;

            /* Compute location of sample in terms of real-valued index array
               coordinates.  Subtract 0.5 so that rx of 1.0 means to put full
               weight on index[1] (e.g., when rpos is 0 and IndexSize is 3.
             */
            rx = rpos + IndexSize / 2.0 - 0.5;
            cx = cpos + IndexSize / 2.0 - 0.5;

            /* Test whether this sample falls within boundary of index patch. */
            if (rx > -1.0 && rx < (float) IndexSize  &&
                    cx > -1.0 && cx < (float) IndexSize)
                AddSample(index, key, grad, ori, irow + i, icol + j, rpos, cpos,
                        rx, cx);
        }
}

//------------------------------------------------------------------------------
void SIFT::AddSample(Float3DArray& index,
        const Key& k, const Image& grad, const Image& orim,
        int r, int c, float rpos, float cpos, float rx, float cx)
{
    float mag, ori, sigma, weight;

    /* Clip at image boundaries. */
    if (r < 0  ||  r >= grad.rows  ||  c < 0  ||  c >= grad.cols)
        return;

    /* Compute Gaussian weight for sample, as function of radial distance
       from center.  Sigma is relative to half-width of index.
     */
    sigma = IndexSigma * 0.5 * IndexSize;
    weight = exp(- (rpos * rpos + cpos * cpos) / (2.0 * sigma * sigma));

    mag = weight * grad.pixels[r][c];

    /* Subtract keypoint orientation to give ori relative to keypoint. */
    ori = orim.pixels[r][c] - k.ori;

    /* Put orientation in range [0, 2*pi].  If sign of gradient is to
       be ignored, then put in range [0, pi].
     */
    if (IgnoreGradSign) {
        while (ori > pi)
            ori -= pi;
        while (ori < 0.0)
            ori += pi;
    } else {
        while (ori > 2*pi)
            ori -= 2*pi;
        while (ori < 0.0)
            ori += 2*pi;
    }
    PlaceInIndex(index, mag, ori, rx, cx);
}

//------------------------------------------------------------------------------
void SIFT::PlaceInIndex(Float3DArray& index,
        float mag, float ori, float rx, float cx)
{
    int r, c, _or, ri, ci, oi, rindex, cindex, oindex;
    float oval, rfrac, cfrac, ofrac, rweight, cweight, oweight;
    float *ivec;

    oval = OriSize * ori / (IgnoreGradSign ? pi : 2*pi);

    ri = int((rx >= 0.0) ? rx : rx - 1.0);  /* Round down to next integer. */
    ci = int((cx >= 0.0) ? cx : cx - 1.0);
    oi = int((oval >= 0.0) ? oval : oval - 1.0);
    rfrac = rx - ri;         /* Fractional part of location. */
    cfrac = cx - ci;
    ofrac = oval - oi;
    assert(ri >= -1  &&  ri < IndexSize  &&  oi >= 0  &&  oi <= OriSize  &&
            rfrac >= 0.0  &&  rfrac <= 1.0);

    /* Put appropriate fraction in each of 8 buckets around this point
       in the (row,col,ori) dimensions.  This loop is written for
       efficiency, as it is the inner loop of key sampling.
     */
    for (r = 0; r < 2; r++) {
        rindex = ri + r;
        if (rindex >=0 && rindex < IndexSize) {
            rweight = mag * ((r == 0) ? 1.0 - rfrac : rfrac);

            for (c = 0; c < 2; c++) {
                cindex = ci + c;
                if (cindex >=0 && cindex < IndexSize) {
                    cweight = rweight * ((c == 0) ? 1.0 - cfrac : cfrac);
                    ivec = &(index[rindex][cindex])[0];

                    for (_or = 0; _or < 2; _or++) {
                        oindex = oi + _or;
                        if (oindex >= OriSize)  /* Orientation wraps around at pi. */
                            oindex = 0;
                        oweight = cweight * ((_or == 0) ? 1.0 - ofrac : ofrac);

                        ivec[oindex] += oweight;
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
void SIFT::TransformLine(Image& im, const Key& k,
        float x1, float y1, float x2, float y2)
{
    int r1, c1, r2, c2;
    float s, c, len;

    /* The scaling of the unit length arrow is set to half the width
       of the region used to compute the keypoint descriptor.
     */
    len = 0.5 * SIFT::IndexSize * SIFT::MagFactor * k.scale;

    /* Rotate the points by k->ori. */
    s = sin(k.ori);
    c = cos(k.ori);

    /* Apply transform. */
    r1 = (int) (k.row + len * (c * x1 + s * y1) + 0.5);
    c1 = (int) (k.col + len * (- s * x1 + c * y1) + 0.5);
    r2 = (int) (k.row + len * (c * x2 + s * y2) + 0.5);
    c2 = (int) (k.col + len * (- s * x2 + c * y2) + 0.5);

    /* Discard lines that have any portion outside of image. */
    if (r1 >= 0 && r1 < im.rows && c1 >= 0 && c1 < im.cols &&
            r2 >= 0 && r2 < im.rows && c2 >= 0 && c2 < im.cols)
        DrawLine(im, r1, c1, r2, c2);
}

//------------------------------------------------------------------------------
void SIFT::WriteKeypoints(FILE *fp, const KeyList& keys)
{
    int i, count = 0;
    KeyList::const_iterator k;

    for (k = keys.begin(); k < keys.end(); ++k)
        count++;

    /* Output total number of keypoints and VecLength. */
    fprintf(fp, "%d %d\n", count, SIFT::VecLength);

    /* Output data for each keypoint. */
    for (k = keys.begin(); k < keys.end(); ++k) {
        fprintf(fp, "%4.2f %4.2f %4.2f %4.3f", k->row, k->col, k->scale,
                k->ori);
        for (i = 0; i < SIFT::VecLength; i++) {
            if (i % 20 == 0)
                fprintf(fp, "\n");
            fprintf(fp, " %d", (int) k->ivec[i]);
        }
        fprintf(fp, "\n");
    }

    /* Write message to terminal. */
    fprintf(stderr, "%d keypoints found.\n", count);
}

//------------------------------------------------------------------------------
// Methods from the former util.c
//------------------------------------------------------------------------------
SIFT::FloatMatrix SIFT::AllocMatrix(int rows, int cols) {
    SIFT::FloatMatrix matrix(rows);
    for (int r = 0;  r < rows;  ++r) {
        matrix[r].resize(cols);
    }
    return matrix;
}

//------------------------------------------------------------------------------
SIFT::Image SIFT::DoubleSize(const Image& image)
{
    int rows, cols, nrows, ncols, r, c, r2, c2;

    Image newimage;

    rows = image.rows;
    cols = image.cols;
    nrows = 2 * rows - 2;
    ncols = 2 * cols - 2;
    newimage.setSize(nrows, ncols);
    const FloatMatrix& im = image.pixels;
    FloatMatrix& _new = newimage.pixels;

    for (r = 0; r < rows - 1; r++)
        for (c = 0; c < cols - 1; c++) {
            r2 = 2 * r;
            c2 = 2 * c;
            _new[r2][c2] = im[r][c];
            _new[r2+1][c2] = 0.5 * (im[r][c] + im[r+1][c]);
            _new[r2][c2+1] = 0.5 * (im[r][c] + im[r][c+1]);
            _new[r2+1][c2+1] = 0.25 * (im[r][c] + im[r+1][c] + im[r][c+1] +
                    im[r+1][c+1]);
        }
    return newimage;
}

//------------------------------------------------------------------------------
SIFT::Image SIFT::HalfImageSize(const Image& image)
{
    int rows, cols, nrows, ncols, r, c, ri, ci;

    Image newimage;

    rows = image.rows;
    cols = image.cols;
    nrows = rows / 2;
    ncols = cols / 2;
    newimage.setSize(nrows, ncols);
    const FloatMatrix& im = image.pixels;
    FloatMatrix& _new = newimage.pixels;

    for (r = 0, ri = 0; r < nrows; r++, ri += 2)
        for (c = 0, ci = 0; c < ncols; c++, ci += 2)
            _new[r][c] = im[ri][ci];
    return newimage;
}

//------------------------------------------------------------------------------
void SIFT::SubtractImage(Image& im1, const Image& im2)
{
    int r, c;

    FloatMatrix& pix1 = im1.pixels;
    const FloatMatrix& pix2 = im2.pixels;

    for (r = 0; r < im1.rows; r++)
        for (c = 0; c < im1.cols; c++)
            pix1[r][c] -= pix2[r][c];
}

//------------------------------------------------------------------------------
void SIFT::GradOriImages(const Image& im, Image& grad, Image& ori)
{
    float xgrad, ygrad;
    int rows, cols, r, c;

    rows = im.rows;
    cols = im.cols;
    const FloatMatrix& pix = im.pixels;

    for (r = 0; r < rows; r++)
        for (c = 0; c < cols; c++) {
            if (c == 0)
                xgrad = 2.0 * (pix[r][c+1] - pix[r][c]);
            else if (c == cols-1)
                xgrad = 2.0 * (pix[r][c] - pix[r][c-1]);
            else
                xgrad = pix[r][c+1] - pix[r][c-1];
            if (r == 0)
                ygrad = 2.0 * (pix[r][c] - pix[r+1][c]);
            else if (r == rows-1)
                ygrad = 2.0 * (pix[r-1][c] - pix[r][c]);
            else
                ygrad = pix[r-1][c] - pix[r+1][c];
            grad.pixels[r][c] = sqrt(xgrad * xgrad + ygrad * ygrad);
            ori.pixels[r][c] = atan2 (ygrad, xgrad);
        }
}

//------------------------------------------------------------------------------
void SIFT::GaussianBlur(Image& image, float sigma)
{
    float x, kernel[100], sum = 0.0;
    int ksize, i;

    /* The Gaussian kernel is truncated at GaussTruncate sigmas from
       center.  The kernel size should be odd.
     */
    ksize = (int)(2.0 * GaussTruncate * sigma + 1.0);
    ksize = LOWE_MAX(3, ksize);    /* Kernel must be at least 3. */
    if (ksize % 2 == 0)       /* Make kernel size odd. */
        ksize++;
    assert(ksize < 100);

    /* Fill in kernel values. */
    for (i = 0; i <= ksize; i++) {
        x = i - ksize / 2;
        kernel[i] = exp(- x * x / (2.0 * sigma * sigma));
        sum += kernel[i];
    }
    /* Normalize kernel values to sum to 1.0. */
    for (i = 0; i < ksize; i++)
        kernel[i] /= sum;

    ConvHorizontal(image, kernel, ksize);
    ConvVertical(image, kernel, ksize);
}

//------------------------------------------------------------------------------
void SIFT::ConvHorizontal(Image& image, float *kernel, int ksize)
{
    int rows, cols, r, c, i, halfsize;
    float buffer[4000];

    rows = image.rows;
    cols = image.cols;
    halfsize = ksize / 2;
    FloatMatrix& pixels = image.pixels;
    assert(cols + ksize < 4000);

    for (r = 0; r < rows; r++) {
        /* Copy the row into buffer with pixels at ends replicated for
           half the mask size.  This avoids need to check for ends
           within inner loop. */
        for (i = 0; i < halfsize; i++)
            buffer[i] = pixels[r][0];
        for (i = 0; i < cols; i++)
            buffer[halfsize + i] = pixels[r][i];
        for (i = 0; i < halfsize; i++)
            buffer[halfsize + cols + i] = pixels[r][cols - 1];

        ConvBufferFast(buffer, kernel, cols, ksize);
        for (c = 0; c < cols; c++)
            pixels[r][c] = buffer[c];
    }
}

//------------------------------------------------------------------------------
void SIFT::ConvVertical(Image& image, float *kernel, int ksize)
{
    int rows, cols, r, c, i, halfsize;
    float buffer[4000];

    rows = image.rows;
    cols = image.cols;
    halfsize = ksize / 2;
    FloatMatrix& pixels = image.pixels;
    assert(rows + ksize < 4000);

    for (c = 0; c < cols; c++) {
        for (i = 0; i < halfsize; i++)
            buffer[i] = pixels[0][c];
        for (i = 0; i < rows; i++)
            buffer[halfsize + i] = pixels[i][c];
        for (i = 0; i < halfsize; i++)
            buffer[halfsize + rows + i] = pixels[rows - 1][c];

        ConvBufferFast(buffer, kernel, rows, ksize);
        for (r = 0; r < rows; r++)
            pixels[r][c] = buffer[r];
    }
}

//------------------------------------------------------------------------------
void SIFT::ConvBuffer(float *buffer, float *kernel, int rsize, int ksize)
{
    int i, j;
    float sum, *bp, *kp;

    for (i = 0; i < rsize; i++) {
        sum = 0.0;
        bp = &buffer[i];
        kp = &kernel[0];

        /* Make this inner loop super-efficient. */
        for (j = 0; j < ksize; j++)
            sum += *bp++ * *kp++;

        buffer[i] = sum;
    }
}

//------------------------------------------------------------------------------
void SIFT::ConvBufferFast(float *buffer, float *kernel, int rsize, int ksize)
{
    int i;
    float sum, *bp, *kp, *endkp;

    for (i = 0; i < rsize; i++) {
        sum = 0.0;
        bp = &buffer[i];
        kp = &kernel[0];
        endkp = &kernel[ksize];

        /* Loop unrolling: do 5 multiplications at a time. */
        while (kp + 4 < endkp) {
            sum += bp[0] * kp[0] +  bp[1] * kp[1] + bp[2] * kp[2] +
                bp[3] * kp[3] +  bp[4] * kp[4];
            bp += 5;
            kp += 5;
        }
        /* Do 2 multiplications at a time on remaining items. */
        while (kp + 1 < endkp) {
            sum += bp[0] * kp[0] +  bp[1] * kp[1];
            bp += 2;
            kp += 2;
        }
        /* Finish last one if needed. */
        if (kp < endkp)
            sum += *bp * *kp;

        buffer[i] = sum;
    }
}

//------------------------------------------------------------------------------
void SIFT::SolveLeastSquares(float *solution, int rows, int cols,
        const FloatMatrix& jacobian,
        float *errvec, FloatMatrix& sqarray)
{
    int r, c, i;
    float sum;

    assert(rows >= cols);
    /* Multiply Jacobian transpose by Jacobian, and put result in sqarray. */
    for (r = 0; r < cols; r++)
        for (c = 0; c < cols; c++) {
            sum = 0.0;
            for (i = 0; i < rows; i++)
                sum += jacobian[i][r] * jacobian[i][c];
            sqarray[r][c] = sum;
        }
    /* Multiply transpose of Jacobian by errvec, and put result in solution. */
    for (c = 0; c < cols; c++) {
        sum = 0.0;
        for (i = 0; i < rows; i++)
            sum += jacobian[i][c] * errvec[i];
        solution[c] = sum;
    }
    /* Now, solve square system of equations. */
    SolveLinearSystem(solution, sqarray, cols);
}

//------------------------------------------------------------------------------
void SIFT::SolveLinearSystem(float *solution, FloatMatrix& sq, int size)
{
    int row, col, c, pivot = 0, i;
    float maxc, coef, temp, mult, val;

    /* Triangularize the matrix. */
    for (col = 0; col < size - 1; col++) {
        /* Pivot row with largest coefficient to top. */
        maxc = -1.0;
        for (row = col; row < size; row++) {
            coef = sq[row][col];
            coef = (coef < 0.0 ? - coef : coef);
            if (coef > maxc) {
                maxc = coef;
                pivot = row;
            }
        }
        if (pivot != col) {
            /* Exchange "pivot" with "col" row (this is no less efficient
               than having to perform all array accesses indirectly). */
            for (i = 0; i < size; i++) {
                temp = sq[pivot][i];
                sq[pivot][i] = sq[col][i];
                sq[col][i] = temp;
            }
            temp = solution[pivot];
            solution[pivot] = solution[col];
            solution[col] = temp;
        }
        /* Do reduction for this column. */
        for (row = col + 1; row < size; row++) {
            mult = sq[row][col] / sq[col][col];
            for (c = col; c < size; c++)	/* Could start with c=col+1. */
                sq[row][c] -= mult * sq[col][c];
            solution[row] -= mult * solution[col];
        }
    }

    /* Do back substitution.  Pivoting does not affect solution order. */
    for (row = size - 1; row >= 0; row--) {
        val = solution[row];
        for (col = size - 1; col > row; col--)
            val -= solution[col] * sq[row][col];
        solution[row] = val / sq[row][row];
    }
}

//------------------------------------------------------------------------------
float SIFT::DotProd(float *v1, float *v2, int len)
{
    int i;
    float sum = 0.0;

    for (i = 0; i < len; i++)
        sum += v1[i] * v2[i];
    return sum;
}

//------------------------------------------------------------------------------
float SIFT::FitQuadratic(float offset[3], const Image *dogs, int s, int r, int c)
{
    float g[3];

    /* First time through, allocate space for Hessian matrix, H. */
    if (m_bUninitializedH) {
        H = AllocMatrix(3, 3);
        m_bUninitializedH = false;
    }

    /* Select the dog images at peak scale, dog1, as well as the scale
       below, dog0, and scale above, dog2.
     */
    const FloatMatrix& dog0 = dogs[s-1].pixels;
    const FloatMatrix& dog1 = dogs[s].pixels;
    const FloatMatrix& dog2 = dogs[s+1].pixels;

    /* Fill in the values of the gradient from pixel differences. */
    g[0] = (dog2[r][c] - dog0[r][c]) / 2.0;
    g[1] = (dog1[r+1][c] - dog1[r-1][c]) / 2.0;
    g[2] = (dog1[r][c+1] - dog1[r][c-1]) / 2.0;

    /* Fill in the values of the Hessian from pixel differences. */
    H[0][0] = dog0[r][c] - 2.0 * dog1[r][c] + dog2[r][c];
    H[1][1] = dog1[r-1][c] - 2.0 * dog1[r][c] + dog1[r+1][c];
    H[2][2] = dog1[r][c-1] - 2.0 * dog1[r][c] + dog1[r][c+1];
    H[0][1] = H[1][0] = ((dog2[r+1][c] - dog2[r-1][c]) -
            (dog0[r+1][c] - dog0[r-1][c])) / 4.0;
    H[0][2] = H[2][0] = ((dog2[r][c+1] - dog2[r][c-1]) -
            (dog0[r][c+1] - dog0[r][c-1])) / 4.0;
    H[1][2] = H[2][1] = ((dog1[r+1][c+1] - dog1[r+1][c-1]) -
            (dog1[r-1][c+1] - dog1[r-1][c-1])) / 4.0;

    /* Solve the 3x3 linear sytem, Hx = -g.  Result gives peak offset.
       Note that SolveLinearSystem destroys contents of H.
     */
    offset[0] = - g[0];
    offset[1] = - g[1];
    offset[2] = - g[2];
    SolveLinearSystem(offset, H, 3);

    /* Also return value of DOG at peak location using initial value plus
       0.5 times linear interpolation with gradient to peak position
       (this is correct for a quadratic approximation).
     */
    return (dog1[r][c] + 0.5 * DotProd(offset, g, 3));
}

//------------------------------------------------------------------------------
SIFT::Image SIFT::ReadPGM(FILE *fp)
{
    int char1, char2, width, height, max, c1, c2, c3, r, c;

    char1 = fgetc(fp);
    char2 = fgetc(fp);
    SkipComments(fp);
    c1 = fscanf(fp, "%d", &width);
    SkipComments(fp);
    c2 = fscanf(fp, "%d", &height);
    SkipComments(fp);
    c3 = fscanf(fp, "%d", &max);

    if (char1 != 'P' || char2 != '5' || c1 != 1 || c2 != 1 || c3 != 1 ||
            max > 255) {
        fprintf(stderr, "ERROR: Input is not a standard raw PGM file.\n"
                "Use xv or PNM tools to convert file to 8-bit PGM format.\n");
        exit(1);
    }
    fgetc(fp);  /* Discard exactly one byte after header. */

    /* Create floating point image with pixels in range [0.0,1.0]. */
    Image image(height, width);
    for (r = 0; r < height; r++)
        for (c = 0; c < width; c++)
            image.pixels[r][c] = ((float) fgetc(fp)) / 255.0;

    return image;
}

//------------------------------------------------------------------------------
SIFT::Image SIFT::fromRawData(char* data, const int width, const int height)
{
  Image image(height, width);
  for (int r = 0; r < height; r++)
      for (int c = 0; c < width; c++)
	  image.pixels[r][c] = ((float) data[r*width+c]) / 255.0;
  return image;
}




//------------------------------------------------------------------------------
void SIFT::SkipComments(FILE *fp)
{
    int ch;

    fscanf(fp," ");      /* Skip white space. */
    while ((ch = fgetc(fp)) == '#') {
        while ((ch = fgetc(fp)) != '\n'  &&  ch != EOF)
            ;
        fscanf(fp," ");
    }
    ungetc(ch, fp);      /* Replace last character read. */
}

//------------------------------------------------------------------------------
void SIFT::WritePGM(FILE *fp, const Image& image)
{
    int r, c, val;

    fprintf(fp, "P5\n%d %d\n255\n", image.cols, image.rows);

    for (r = 0; r < image.rows; r++)
        for (c = 0; c < image.cols; c++) {
            if (image.pixels[r][c] == 2.0)
                fputc(255, fp);
            else {
                val = (int) (255.0 * image.pixels[r][c]);
                if ((val >= 250) && (val <= 255))
                    fputc(250, fp);
                else
                    fputc(LOWE_MAX(0, LOWE_MIN(255, val)), fp);
            } // if - else
        }
}

//------------------------------------------------------------------------------
void SIFT::DrawLine(Image& image, int r1, int c1, int r2, int c2)
{
    int i, dr, dc, temp;

    if (r1 == r2 && c1 == c2)  /* Line of zero length. */
        return;

    /* Is line more horizontal than vertical? */
    if (SIFT::LOWE_ABS(r2 - r1) < SIFT::LOWE_ABS(c2 - c1)) {

        /* Put points in increasing order by column. */
        if (c1 > c2) {
            temp = r1; r1 = r2; r2 = temp;
            temp = c1; c1 = c2; c2 = temp;
        }
        dr = r2 - r1;
        dc = c2 - c1;
        for (i = c1; i <= c2; i++)
            image.pixels[r1 + (i - c1) * dr / dc][i] = 2.0;

    } else {

        if (r1 > r2) {
            temp = r1; r1 = r2; r2 = temp;
            temp = c1; c1 = c2; c2 = temp;
        }
        dr = r2 - r1;
        dc = c2 - c1;
        for (i = r1; i <= r2; i++)
            image.pixels[i][c1 + (i - r1) * dc / dr] = 2.0;
    }
}

