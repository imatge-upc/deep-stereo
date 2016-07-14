function rgb = yuv2rgb(yuv,spacein);
% YUV2RGB  Converts PAL YUV images in Colour Space to RGB components
%
%   rgb = YUV2RGB(yuv) converts color image YUV to RGB.
%   It needs a yuv in the range of [0,1]. (u and v [-0.5,0.5]) 
%   The rgb output will be clipped. 
%
%   See also RASTERREAD, RASTERYUVREAD, RGB2YUV.
%
%
%               Javier Ruiz Hidalgo (UPC) (18/01/99).

% Revision History
%   0.1.2 - Change conversion tables completely, it now follows the
%           YUV Colour Space 
%   0.1.1 - Changed slightly some of the conversion functions. 
%   0.1   - Started coding from rgb2yuv function - (18/01/99).


  
% Check input args
  if (nargin<=1),
	space = 'yuv';
  else
	space = spacein;
  end;
  
  % Get proper space values  
  switch lower(space),
	
   case 'yuv?',
	cry = 1; cru = 0;      crv = 1.14;
	cgy = 1; cgu = -0.395; cgv = -0.581;
	cby = 1; cbu = 2.032;  cbv = 0;
	
   case 'pal',
	cry = 1; cru = 0;      crv = 1;
	cgy = 1; cgu = -0.194; cgv = -0.509;
	cby = 1; cbu = 1;      cbv = 0;
	
   case 'yuv',
	cry = 1; cru = 0;      crv = 1.371;
	cgy = 1; cgu = -0.336; cgv = -0.698;
	cby = 1; cbu = 1.732;  cbv = 0;
	
  end;
  

  % Check image 
  y = yuv(:,:,1);
  u = yuv(:,:,2);
  v = yuv(:,:,3);
  
  if (min(min(min(y)))<0.0) | (max(max(max(y)))>1.0) ...
  		| (min(min(min(u)))<-0.5) | (max(max(max(u)))>0.5) ...
		| (min(min(min(v)))<-0.5) | (max(max(max(v)))>0.5),
	
	error(['YUV image must be in the range Y:[0,1] and UV:[-0.5,' ...
		   ' 0.5]']);
  end;
  
  r = cry*y + cru*u + crv*v;
  g = cgy*y + cgu*u + cgv*v;
  b = cby*y + cbu*u + cbv*v;
  
  rgb = cat(3,r,g,b);
	  
  % There can be some problems converting to rgb as round values might 
  % set rgb values over the [0,1] range.
  idx = find(rgb<0); rgb(idx) = 0;
  idx = find(rgb>1); rgb(idx) = 1;
  
  
