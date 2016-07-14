function [seqsize,fstart,fend,fps,type] = yuvreaddat (datname)
%
% YUVREADDAT reads .dat file from yuv sequences 
%
%   Usage: [seqsize,ftart,fend,fps,type] = YUVREADDAT (datname) where:
%      datname : input dat name of the yuv file (with extension)
%
%   So for example use:
%      yuvreaddat ('seq.dat');
%
%           Javier Ruiz Hidalgo <jrh@gps.tsc.upc.edu>

% Read .dat file
%dat = load(datname);
fid = fopen(datname);
tline = fgetl(fid);
fclose(fid);

% Convert to vector
dat = sscanf(tline,'%d %d %d %f %d',5);
        
% Fill output variables
seqsize = [dat(1) dat(2)];
fstart = 0;
fend   = dat(3) - 1;
fps    = dat(4);
type   = dat(5);
