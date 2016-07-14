function yuvshow (inname,f)
%
% IMGSHOW displays yuv files
%
%   Usage: YUVSHOW (inname,f) where:
%      inname : input name of yuv file (without extension .yuv)
%      f      : NUmber of frames (if ommited all are shown)
%
%           Javier Ruiz Hidalgo <j.ruiz@upc.edu>



[seqsize,ffstart,ffend,ffps,type] = yuvreaddat([inname '.dat']);

% Check input args
if (nargin<2),
	fstart = ffstart;
    fend = ffend;
else
	fstart = f;
    fend = f;    
end;


% For each frame
for i=fstart:fend,
  
	% Read yuv image
    [y,u,v] = yuvread(inname,i);
	% Resize if needed
 	if (type==420), 
		% imresize U and V
		u2 = imresize (u,2);
        v2 = imresize (v,2);
	else,
		u2 = u;
		v2 = v;
	end;  

    yuv(:,:,1) = y/255;
    yuv(:,:,2) = (u2-128)/255;
    yuv(:,:,3) = (v2-128)/255;
  
    ff = yuv(:,:,2);    
    idx = find(ff<-0.5); ff(idx) = -0.5;
    idx = find(ff>0.5);  ff(idx) = 0.5;
    yuv(:,:,2) = ff;    

    ff = yuv(:,:,3);    
    idx = find(ff<-0.5); ff(idx) = -0.5;
    idx = find(ff>0.5);  ff(idx) = 0.5;
    yuv(:,:,3) = ff;    

	% Convert to RGB
	rgb = yuv2rgb(yuv);
   
	% Show RGB image
    i,
    imshow(rgb);
    pause;  
end;
