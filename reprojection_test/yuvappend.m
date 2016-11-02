function yuvappend (name, y,u,v)
% YUVAPPEND appends an image (YUV 8 bits per pixel) to a sequence
%
%   Usage: 
%      YUVAPPEND (name, y, u, v) Appends YUV components to a sequence
%      make sure the dinamic range is [0,255].
%
%      YUVAPPEND (name,y) appends 400 sequences.
%
%          Javier Ruiz Hidalgo <jrh@gps.tsc.upc.es>, UPC, 08/09/00
%


  % Open the file for appending
  fid = fopen(name, 'a');
  if (fid==-1),
	 error('Couldn''t open file.');
  end;

  % Append components
  fwrite( fid, uint8(y'), 'uchar');
  if (nargin>2),
	  fwrite( fid, uint8(u'), 'uchar');
	  fwrite( fid, uint8(v'), 'uchar');
  end;

  % Close the file
  fclose(fid);













