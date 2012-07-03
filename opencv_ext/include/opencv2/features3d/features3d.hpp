/*
 * features3d.h
 *
 *  Created on: May 2, 2012
 *      Author: David Gossow
 */

#ifndef FEATURES3D_H_
#define FEATURES3D_H_

#include "keypoint3d.hpp"
#include <opencv2/core/core.hpp>

namespace cv
{

/*
 * Abstract base class for 2D image feature detectors.
 */
class CV_EXPORTS FeatureDetector3D
{
public:
    virtual ~FeatureDetector3D();

    /*
     * Detect keypoints in an image.
     * image        The image.
     * keypoints    The detected keypoints.
     * mask         Mask specifying where to look for keypoints (optional). Must be a char
     *              matrix with non-zero values in the region of interest.
     */
    void detect( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const;

    /*
     * Detect keypoints in an image set.
     * images       Image collection.
     * keypoints    Collection of keypoints detected in an input images. keypoints[i] is a set of keypoints detected in an images[i].
     * masks        Masks for image set. masks[i] is a mask for images[i].
     */
    void detect( const vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, const vector<Mat>& masks=vector<Mat>() ) const;

    // Read detector object from a file node.
    virtual void read( const FileNode& );
    // Read detector object from a file node.
    virtual void write( FileStorage& ) const;

    // Return true if detector object is empty
    virtual bool empty() const;

    // Create feature detector by detector name.
    static Ptr<FeatureDetector> create( const string& detectorType );

protected:
    virtual void detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat() ) const = 0;

    /*
     * Remove keypoints that are not in the mask.
     * Helper function, useful when wrapping a library call for keypoint detection that
     * does not support a mask argument.
     */
    static void removeInvalidPoints( const Mat& mask, vector<KeyPoint>& keypoints );
};

}

#endif /* FEATURES3D_H_ */
