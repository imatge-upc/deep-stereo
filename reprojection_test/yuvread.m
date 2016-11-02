function [Y,U,V] = yuvread( name, frame, size, type)
% YUVREAD reads an image from a YUV 8 bits per pixel sequence
%
%   Returns the Y U V images read
%
%   Usage: 
%      [y,u,v] = YUVREAD ('name', frame) reads the filename 'name.yuv' 
%      skipping the first 'frame-1' (first is 0) images. 'name.dat' 
%      will be read to obtain the size and type of the sequence.
%
%      [y,u,v] = YUVREAD ('name', frame, size) reads the filename 'name' 
%      skipping the first 'frame-1' (first is 0) images. 'size' is a 
%      vector with [176 144] for qcif, etc...
%
%      [y,u,v] = YUVREAD ('name',frame,size,type) with 'type' an number
%      indicating the YUV type. Supported 444,422,420 and 400 (only Y).
%      Defaults to 420.
%      
%
%          Javier Ruiz Hidalgo <jrh@gps.tsc.upc.es>, UPC, 08/09/00
%

if (nargin==2),
	seqname = [name '.yuv'];
	[size,fstart,fend,fps,type] = yuvreaddat ([name '.dat']);
else
	seqname = name;
end;

if (nargin==3),
	type = 420;
end;


% Open the file for reading
fid = fopen(seqname, 'r');
if (fid==-1),
	error('Couldn''t open file.');
end;

  % Compute skip size
  y_size = size(1)*size(2);
  switch (type),
    case 444,
      uv_size = y_size;
      size_uv = size;
    case 422,
      uv_size = y_size/2;
      size_uv = [size(1)/2 size(2)];
    case 420,
      uv_size = y_size/4;
      size_uv = size/2;
    case 400,
      uv_size = 0;
      size_uv = 0;
  end;
  
  skip = y_size + (2*uv_size);

  % Skip previous frames
  skip = frame * skip;
  status = fseek( fid, skip, 'bof');
  if (status==-1),
	 error(['Couldn''t find desire frame: ' ferror(fid)]);
  end;
  
  
  % Read Y
  [Y,cnt] = fread( fid, size, 'uchar');
  if (cnt~=y_size),
	 error('Couldn''t read Y image.');
  end

  if (type~=400),

    % Read U
    [U,cnt] = fread( fid, size_uv, 'uchar');
    if (cnt~=uv_size),
      error('Couldn''t read U image.');
    end


    % Read V 
    [V,cnt] = fread( fid, size_uv, 'uchar');
    if (cnt~=uv_size),
	   error('Couldn''t read V image.');
    end

  end;

  % Flip and rotate images
  Y = Y'; %imrotate(fliplr(Y),90);
  if (type~=400),
    U = U'; %imrotate(fliplr(U),90);
    V = V'; %imrotate(fliplr(V),90);
  end;

  % Close the file
  fclose(fid);

