/*
  Copyright (C) 2009-2012 Fraunhofer SCAI, Schloss Birlinghoven, 53754 Sankt Augustin, Germany;
  all rights reserved unless otherwise stated.
  
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
  MA 02111-1307 USA
*/

class GPUTimer {

  public:

   GPUTimer() {
     cudaEventCreate(&evStart, 0);
     cudaEventCreate(&evStop, 0);
     time = 0.0;
   }

   void start() {
     cudaEventRecord(evStart);
     cudaEventSynchronize(evStart);
   }

   void stop() {
     cudaEventRecord(evStop);
     cudaEventSynchronize(evStop);
     cudaEventElapsedTime(&time1, evStart, evStop);
     time += time1;
   }

   float get() const {
     return time;
   }

   float last() const {
     return time1;
   }

  private:

   float time;
   float time1;

   cudaEvent_t evStart;
   cudaEvent_t evStop;

};

