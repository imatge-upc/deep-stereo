function yuvoverwrite (outname,force)

if (nargin<2),
  force = 'no';
end;

if (exist ([outname '.yuv'],'file')>0),
  if (strncmp(force,'yes',2)),
    delete ([outname '.yuv']);
  else,
    a = input (['File ' outname '.yuv exists. Delete file (y/n): '],'s');
    if (a=='y'),
      delete ([outname '.yuv']);
    else
      error ('File exists');
    end;
  end;
end;
if (exist ([outname '.dat'],'file')>0),
    delete ([outname '.dat']);
end;
