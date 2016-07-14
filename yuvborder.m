function yuvdecimate (inname,outname,border)
%
% YUVBORDER adds/removes borders to .yuv files 
%
%   Usage: YUVBORDER (inname, outname,border) where:
%      inname  : input name of the yuv file (without extension and .dat needed)
%      outname : output name of yuv file (without extension, .dat file will be created)
%      border  : border size in pixels (if negative it will be removed)
%
%           Javier Ruiz Hidalgo <j.ruiz@upc.edu>

% Remove last yuv as yuvappend does not do it
yuvoverwrite (outname);

% Read .dat file
[insize,fstart,fend,fps,type] = yuvreaddat ([inname '.dat']);

for i=fstart:fend,
  
  i,
  
  [y,u,v] = yuvread ([inname '.yuv'], i, insize);
  
  % Resize images
  if (border<=0),
    b = -border;
    y2 = y(1+b:end-b,1+b:end-b);
    switch type
        case 444
          u2 = u(1+b:end-b,1+b:end-b);
          v2 = v(1+b:end-b,1+b:end-b);
        case 420
          if mod(b,2)==1,
              error('Border must be divisible by 2 in 420 sequences');
          end 
          b = b/2;
          u2 = u(1+b:end-b,1+b:end-b);
          v2 = v(1+b:end-b,1+b:end-b);
        otherwise
          error('Unknown sequence type')
    end;
  end;
  
  % Resize images
  if (border>0),
    error('Not implemented');
    b = [border border];
    y2 = padarray(y,b);
    switch type
        case 444
          u2 = padarray(u,b);
          v2 = padarray(v,b);
        case 420
          if mod(border,2)==1,
              error('Border must be divisible by 2 in 420 sequences');
          end 
          b = [border/2 border/2];
          u2 = padarray(u,b);
          v2 = padarray(v,b);
        otherwise
          error('Unknown sequence type')
    end;

  end
  
  yuvappend ([outname '.yuv'],y2,u2,v2);
  
end;

% Create .dat file
fid = fopen([outname '.dat'],'w');
fprintf(fid,'%d %d %d %2.1f %d\n',size(y2,2), size(y2,1), fend-fstart+1, fps, type);
fclose(fid);

