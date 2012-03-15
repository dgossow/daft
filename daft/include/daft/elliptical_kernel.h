/*
* Copyright (C) 2011 David Gossow
*/

#include <opencv2/core/core.hpp>

#include <iostream>

#include "math_stuff.h"

#ifndef rgbd_features_anisotropic_gauss_h_
#define rgbd_features_anisotropic_gauss_h_

namespace cv
{
namespace daft
{


/**
 * Approximate ellipse by a convex polygon with
 * multiples of 45° as tangential angles
 * @param beta,b,a: angle + major/minor axis length
 * @param vertices: first 4 of the 8 outer vertices of the polygon
*/
void approxEllipse45( float beta, float b, float a, Point2f vertices[4] )
{
  float cos_beta = cos(beta);
  float sin_beta = sin(beta);

  // these vectors characterise the
  // points on the ellipse we're looking for

  Point2f v[4];

  v[0].x = a*cos_beta;
  v[0].y = b*sin_beta;

  v[1].x = a*sin_beta;
  v[1].y = b*cos_beta;

  v[2].x = a*(sin_beta-cos_beta);
  v[2].y = b*(sin_beta+cos_beta);

  v[3].x = a*(sin_beta+cos_beta);
  v[3].y = b*(sin_beta-cos_beta);

  // compute the points where the ellipse has a tangent of
  // 0°, 45°, 90° and 135°.

  Point2f n[4];

  for ( int i=0; i<4; i++ )
  {
    alpha = 2 * atan( (-v[i].x + norm(v[i]) ) / v[i].y );
    cos_alpha = cos(alpha);
    sin_alpha = sin(alpha);
    n[i].x = a*cos_alpha*cos_beta + b*sin_alpha*sin_beta;
    n[i].y = a*cos_alpha*sin_beta + b*sin_alpha*cos_beta;
  }

  // compute intersection of every pair of consecutive tangents
  Point2f inter[5];

  inter[0].x = n[0].x;
  inter[0].y = n[1].y - n[1].x +n[0].x;

  inter[1].x = n[1].x - n[1].y +n[2].y;
  inter[1].y = n[2].y;

  inter[2].x = n[3].x + n[3].y -n[2].y;
  inter[2].y = n[2].y;

  inter[3].x = -n[0].x;
  inter[3].y = n[3].y - n[3].x -n[0].x;

  // the fifth intersection is already 180° from the first
  inter[4] = -inter[0];

  // compute area of approximated ellipse,
  // which is composed of the triangles from
  // center to consecutive intersections
  float area_poly=0;

  for ( int i=0; i<4; i++ )
  {
    float l1 = norm( inter[i] );
    float l2 = norm( inter[i+1] );
    float l3 = norm( inter[i+1] - inter[i] );
    area_poly += sqrt( (l1+l2-l3) * (l1-l2+l3) * (-l1+l2+l3) * (l1+l2+l3) );
  }

  float area_ellipse = M_PI * a * b;
  float scaling = sqrt( area_poly / area_ellipse );

  for ( int i=0; i<4; i++ )
  {
    vertices[i] = inter[i] * scaling;
  }
}

#define ASTEP 16;
#define MSTEP 16;

Point2f ellipse_vertices[ASTEP][MSTEP][4];

void computeEllipticalKernels( )
{
  for ( int a = 0; a < ASTEP; a++ )
  {
    for ( int m = 0; m < MSTEP; m++ )
    {
      float angle = float(a) / float(ASTEP) * M_PI;
      float minor = float(m) / float(MSTEP-1);
      approxEllipse45( angle, 1, minor, ellipse_vertices[a][m] );
    }
  }
}

float ellipseMean( const Mat1d& ii, const Mat1d& lii, const Mat1d& rii, float x, float y, float angle, float major, float minor )
{
  int a = ( angle * float(ASTEP) / M_PI + 0.5f ) % ASTEP;
  int m = minor * float(MSTEP-1);
  Point2f vertices[4] = ellipse_vertices[a][m];
  float i1 = ii[ y + int( vertices[0] ) ][];
}

}}
