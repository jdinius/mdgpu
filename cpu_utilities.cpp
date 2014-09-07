//
//  FILE:   utilities.cpp
//  MODULE: diskDynamics
//
//  DESCRIPTION:
//  File contains all of the routines for running generalized collision simulations.
//
//  REFERENCE:
//  Dinius, J.  "Module Description of diskDynamics, version 1.0".
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13   
//

// INCLUDES/DECLARATIONS
#include "utilities.h" // <see this file for global variable descriptions>
//#include "math.h"
#include <iostream>
#include "stdlib.h"
#include <cmath>

using namespace std;

// END INCLUDES/DECLARATIONS

//
//  FUNCTION: hardStep
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//  Function performs one-time step integration/iteration of hard disk system and stores velocity values for velocity autocorrelation processing.
//
//  INPUTS:
//  t - time step (updated throughout the call; interpreted as the time remaining on the current step)
//
//  OUTPUTS:
//  (none)
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  Time
//  vacfSettleTime
//  vacfCaptureTime
//  modVacf
//  noColl
//  maxFlight
//  DIM
//  nDisks
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  iVacf  (incremented)
//  i_coll (incremented)
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
void hardStep(double t)
{
    COLL c;     // instance of COLL containing next collision states (index, partner and time)
    
    // DECREMENT TIME t UNTIL IT IS EQUAL TO 0
    while (t > 0.0){
        // FIND NEXT COLLISION AND TUMBLE
        c = nextColl();
        
        // FIND MINIMUM TIME OF NEXT EVENT RELATIVE TO THE CURRENT TIME
        
        if (t < c.time){
            // IF t IS SMALLER THAN THE TIMES OF NEXT TUMBLE AND COLLISION, PROPAGATE THE POSITIONS FORWARD WITH AN EULER INTEGRATION STEP
            freeFlight(t);
            
            // ENFORCE PERIODIC BOUNDARY CONDITIONS
            boxSet();
            
            // UPDATE COLLISION TIMES FOR ALL DISKS
            updateTimes(t, noColl, noColl);
            
            // SET t TO ZERO, INDICATING CURRENT CALL TO hardStep IS COMPLETE
            t = 0.0;
        } // END if (t < tm)
        else {
            // COLLISION EVENT IS NEXT
            
            // PROPAGATE DISK POSITIONS FORWARD BY TIME OF NEXT COLLISION
            freeFlight(c.time);
            t -= c.time;
            
            // ENFORCE PERIODIC BOUNDARY CONDITIONS
            boxSet();
            
            if (c.partner != maxFlight){
                // IF THE NEXT COLLISION EVENT OCCURS BETWEEN TWO DISKS (NOT A MAX-FLIGHT CONDITION), UPDATE PHASE SPACE AND TANGENT VECTORS AFTER COLLISION
                collision(c.index,c.partner);
                
                // INCREMENT COLLISION COUNTER
                ++i_coll;
            } // END if (c.partner != maxFlight)
            
            
            // UPDATE COLLISION TIMES FOR ALL DISKS
			updateTimes(c.time, c.index, c.partner);
			//updateTimes(c.time, c.partner, c.index);
			/*if (c.index < c.partner){
				updateTimes(c.time, c.index, c.partner);
			}
			else {
				updateTimes(c.time, c.partner, c.index);
			}*/
        } // END else if (c.time < tmb.time)
        
    } // while (t > 0.0)
    
    return;
}

//
//  FUNCTION: nextColl
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//  Function returns parameters for the next collision based upon the current sim time.  The next collision is determined through a simple minimizing search.
//
//  INPUTS:
//  (none)
//
//  OUTPUTS:
//  c - COLL structure containing the next colliding disk's index, partner and the associated collision time (relative to current sim time)
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  collArray
//  nDisks
//  bigTime
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  (none)
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
COLL nextColl(void)
{
    int i;            // loop index
    COLL c;           // instance of COLL containing next collision states (index, partner and time) 
    c.time = bigTime; // initialize next collision time to something large
    
    for (i = 0; i < nDisks; i++) {
        // FOR ALL DISKS
        if ( (collArray[i].time < c.time) && (collArray[i].time > 0.0) ){
            // IF CURRENT VALUE FOR NEXT COLLISION TIME IS LARGER THAN THE VALUE CONTAINED IN collArray (WITH INDEX i) AND THAT VALUE IS LARGER THAN 0.0 (INDICATING THE COLLISION TIME IS VALID), THE NEXT COLLISION TIME, INDEX, AND PARTNER ARE UPDATED WITH VALUES FROM collArray (WITH INDEX i)
            c = collArray[i];
            
        } // END for (i = 0; i < nDisks; i++
        
    } // END if ( (collArray[i].time < c.time) && (collArray[i].time > 0.0) )
    
    // RETURN GLOBAL MINIMIZER OF NEXT COLLISION TIME, AND THE ASSOCIATED INDEX AND PARTNER
    return c;
}

//
//  FUNCTION: freeFlight
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//  Function updates state and tangent space offset vectors during the free-flight portion of the dynamics by an Euler integration step.
//
//  INPUTS:
//  dt - Euler integration timestep
//
//  OUTPUTS:
//  (none)
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  nlya
//  DIM
//  nDisks
//  phaseDim
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  y (positions integrated forward in time, velocities left unchanged)
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
void freeFlight(double dt)
{
    int i,j; // loop indices
    for (i = 0; i < nlya+1; i++){
        // FOR EACH OF THE COMBINED STATE AND TANGENT VECTORS
        for (j = 0; j < DIM*nDisks; j++){
            // UPDATE THE POSITION COMPONENTS WITH THE TIME-INTEGRATED MOMENTUM COMPONENTS
            y[i*phaseDim+j] += y[DIM*nDisks+i*phaseDim+j]*dt;
        } // END for (j = 0; j < DIM*nDisks; j++)
        
    } // END for (i = 0; i < nlya+1; i++)
    
    return;
}

//
//  FUNCTION: boxSet
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//  Function applies periodic boundary conditions to the position components of the state vector to ensure all disks remain in the simulation box at all times.
//
//  INPUTS:
//  (none)
//
//  OUTPUTS:
//  (none)
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  boxSize
//  nDisks
//  DIM
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  y (positions updated according to periodic boundary conditions)
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
void boxSet(void)
{
    int i,j; // loop indices
    
    for (i = 0; i < nDisks; i++){
        // FOR ALL DISKS
        for (j = 0; j < DIM; j++){
            // FOR EACH SPATIAL COMPONENT (X AND Y)
            while ( y[i*DIM+j] < 0.0 ){
                // WHILE THE PARTICLE POSITION IS LEFT (X-DIR) OR BELOW (Y-DIR) THE BOX SIZE IN THE CURRENT DIMENSION (boxSize[j]), ADD boxSize[j] UNTIL THE VALUE IS POSITIVE AND LESS THAN boxSize[j]
                y[i*DIM+j] += boxSize;
            } // END while ( y[i*DIM+j] < 0.0 )
            
            while ( y[i*DIM+j] > boxSize ){
                // WHILE THE PARTICLE POSITION IS RIGHT (X-DIR) OR ABOVE (Y-DIR) THE BOX SIZE IN THE CURRENT DIMENSION (boxSize[j]), SUBTRACT boxSize[j] UNTIL THE VALUE IS POSITIVE AND LESS THAN boxSize[j]
                y[i*DIM+j] -= boxSize;
            } // END while ( y[i*DIM+j] > boxSize[j] )
            
        }// END for (j = 0; j < DIM; j++)
        
    } // END for (i = 0; i < nDisks; i++)
    
    return;
}

//
//  FUNCTION: updateTimes
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//  Function updates the collision and tumble times of all disks and stores them in the collArray and tumbleArray structures, respectively.  This routine is called to update the collision times and partners for each disk after a collision has occurred.
//
//  INPUTS:
//  tLast    - time states have been propagated forward since the last call
//  isTmbl   - flag indicating whether or not a tumble is to occur in the 
//             current call of this function
//  d1       - index of first disk undergoing collision
//  d2       - index of second disk undergoing collision
//  tmblIndx - index of disk undergoing tumble
//
//  OUTPUTS:
//  (none)
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  nDisks
//  maxFlight
//  noColl
//  maxFlightDblArray
//  dtTmbl
//  DIM
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  collArray   (updated values for collision partners and associated times for all nDisks disks)
//  tumbleArray (updated values for tumble times for all nDisks disks)
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
void updateTimes(double tLast, int d1, int d2)
{
    int i,j;   // loop indices
    double t1; // temporary variable for comparisons
    
    // UPDATE COLLISION AND TUMBLE TIMES FROM THOSE COMPUTED DURING LAST FUNCTION CALL
    for (i = 0; i < nDisks; i++){
        collArray[i].time   -= tLast;
    } // END for (i = 0; i < nDisks; i++)
    
	// UPDATE COLLISION TIMES (RELATIVE TO CURRENT SIM TIME)
	if ( d1 != noColl ){
		// IF A COLLISION HAS OCCURRED (noColl IS USED TO UPDATE COLLISION TIMES AFTER A FREE-FLIGHT)
		for (i = 0; i < nDisks; i++){
			// FOR ALL DISKS
			if ( (i == d1) || // IF i IS EQUAL TO d1
				(collArray[i].partner == d1) || // OR THE COLLISION PARTNER OF i IS d1
				(i == d2) || // OR i IS EQUAL TO d2
				(collArray[i].partner == d2) ) { // OR THE COLLISION PARTNER OF i IS d2

					// IF ONE OF THE ABOVE CONDITIONS, DISK i WAS INVOLVED IN THE LAST COLLISION

					// RESET THE TIME OF NEXT COLLISION TO A LARGE NUMBER AND SET THE PARTNER TO BE maxFlight
					collArray[i].time    = bigTime;
					collArray[i].partner = maxFlight;

					// NOW SEARCH FOR THE MINIMUM TIME AND THE ASSOCIATED PARTNER OF THE NEXT COLLISION FOR DISK i
					for (j = 0; j < nDisks; j++){
						if (i != j){
							// LOOP OVER ALL DISKS (that are not identical to disk i)
							t1 = binTime(i,j);

							if (t1 > 0.0) {
								// IF COLLISION TIME IS POSITIVE, THEN THE COLLISION IS A VALID ONE
								if (t1 < collArray[i].time) {
									// IF THE COLLISION TIME BETWEEN DISKS j AND i IS SMALLER THAN THE CURRENT CALCULATED COLLISION TIME FOR DISK i, UPDATE THE TIME AND PARTNER FOR DISK i
									collArray[i].time = t1;
									collArray[i].partner = j;
								} // END if (t1 < collArray[i].time)

								if (t1 < collArray[j].time) {
									// IF THE COLLISION TIME BETWEEN DISKS j AND i IS SMALLER THAN THE CURRENT CALCULATED COLLISION TIME FOR DISK j, UPDATE THE TIME AND PARTNER FOR DISK j
									collArray[j].time = t1;
									collArray[j].partner = i;
								} // END if (t1 < collArray[j].time)

							} // END if (t1 > 0.0)

						} // END if (i != j)
					} // END for (j = 0; j < nDisks; j++)

			} // END if ( (i == d1) || ...

		} // END for (i = 0; i < nDisks; i++)

	} // END if ( d1 != noColl )
    
    // CHECK FOR MAX FLIGHT CONDITION
    for (i = 0; i < nDisks; i++){
        // FOR ALL DISKS
        for (j = 0; j < DIM; j++){
            // FOR BOTH SPATIAL DIMENSIONS
            if ( fabs(collArray[i].time * y[DIM*(i+nDisks)+j]) > maxFlightDblArray[j] ){
                // IF THE DISTANCE TRAVELED IN EITHER SPATIAL DIRECTION BEFORE A VALID COLLISION INVOLVING DISK i EXCEEDS THE MAXIMUM FLIGHT CONDITION maxFlight, THEN SET THE COLLISION TIME AND PARTNER FOR DISK i ACCORDINGLY
                collArray[i].time    = fabs(maxFlightDblArray[j] / y[DIM*(i+nDisks)+j]);
                collArray[i].partner = maxFlight;
            }// END if ( fabs(collArray[i].time * y[DIM*(i+nDisks)+j]) > maxFlightDblArray[j] )
            
        } // END for (j = 0; j < DIM; j++)
        
    } // END for (i = 0; i < nDisks; i++)
    
    return;
}

//
//  FUNCTION: binTime
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//  Function finds time when particles i1 and j1 undergo collision.  Function returns -1 if collision between the two disks is impossible.
//
//  INPUTS:
//  i1 - index of first disk
//  j1 - index of second disk
//
//  OUTPUTS:
//  t1 - collision time relative to current sim time (-1 if no valid collision was found)
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  DIM
//  y
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  yi (dummy variable)
//  yj (dummy variable)
//  vi (dummy variable)
//  vj (dummy variable)
//  dq1 (dummy variable, used to store shortest distance between disk centers)
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
double binTime(int i1, int j1)
{
    int ii;    // loop index
    double t1; // return value
    
    // DECLARE VECTORS FOR CONVENIENCE
    for (ii = 0; ii < DIM; ii++){
        yi[ii] = y[DIM*i1+ii]; // position of ith particle
        yj[ii] = y[DIM*j1+ii]; // position of jth particle
        vi[ii] = y[DIM*(i1+nDisks)+ii]; // momentum of ith particle
        vj[ii] = y[DIM*(j1+nDisks)+ii]; // momentum of jth particle
    } // END for (ii = 0; ii < DIM; ii++)
    
    // FIND SHORTEST DISTANCE BETWEEN DISKS i AND j CENTERS-OF-MASS
    image(yi,yj);
    
    // INITIALIZE delta
    double delta = 0.0;
    
    // COMPUTE delta EQUALS DOT PRODUCT OF RELATIVE POSITION VECTOR WITH RELATIVE MOMENTUM VECTOR
    for (ii = 0; ii < DIM; ii++){
        delta += dq1[ii] * (vi[ii]-vj[ii]);
    } // END for (ii = 0; ii < DIM; ii++)
    
    if (delta>0) {
        // IF THE DOT PRODUCT IS POSITIVE, THEN DISKS ARE MOVING AWAY FROM EACH OTHER AND NO COLLISION IS POSSIBLE
        t1 = -1.0;
        return t1;
    } // END if (delta>0)
    
    // INITIALIZE dq2 and dv2
    double dq2 = 0.0;
    double dv2 = 0.0;
    
    // COMPUTE dq2 EQUALS DOT PRODUCT OF RELATIVE POSITION VECTOR WITH ITSELF AND dv2 EQUALS DOT PRODUCT OF RELATIVE MOMENTUM VECTOR 
    // WITH ITSELF
    for (ii = 0; ii < DIM; ii++){
        dq2 += pow(dq1[ii],2.0);
        dv2 += pow( (vi[ii]-vj[ii]), 2.0 );
    } // END for (ii = 0; ii < DIM; ii++)
    
    // COMPUTE DISCRIMINANT IN QUADRATIC EQUATION FOR COLLISION TIME; 1.O FACTOR COMES FROM ASSUMPTION THAT EACH DISK DIAMETER IS 1.0
    double discr = pow(delta,2.0) - dv2*(dq2 - 1.0);
    
    if (discr < 0){
        // IF THE DISCRIMINANT IS LESS THAN ZERO, THEN THE SQUARE ROOT OF THE DISCRIMINANT WILL BE COMPLEX INDICATING THAT NO COLLISION IS POSSIBLE
        t1 = -1.0;
        return t1;
    } // END if (discr < 0)
    
    // COMPUTE COLLISION TIME OF DISKS i AND j AS THE "-" SOLUTION OF QUADRATIC EQUATION (IT IS THE SOONER OF THE TWO TIMES)
    t1 = (-delta - sqrt(discr)) / dv2;
    
    return t1;
}

//
//  FUNCTION: image
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//  Function outputs vector of shortest distance between two xy positions in the simulation box
//
//  INPUTS:
//  y1 - xy position of first disk
//  y2 - xy position of second disk
//
//  OUTPUTS:
//  (none)
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  DIM
//  boxSize
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  dq1 (computed vector of shortest length connecting positions y1 and y2 incorporating periodic boundary
//       conditions)
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
void image(double *y1,double *y2)
{
    int i; // loop index
    for (i = 0; i < DIM; i++){
        // FOR EACH SPATIAL DIRECTION, COMPUTE THE RELATIVE POSITION COMPONENT ALONG THE CURRENT DIRECTION
        dq1[i] = y1[i] - y2[i];
        // THE SHORTEST DISTANCE BETWEEN TWO DISKS IN THE BOX CAN BE NO GREATER THAN 1/2 THE BOX LENGTH IN EITHER DIRECTION (X OR Y)
        if (dq1[i] > boxSize / 2.0){
            // POSITIVE CHECK
            dq1[i] -= boxSize;
        } // END if (dq1[i] > boxSize[i] / 2.0)
        
        else if (dq1[i] < -boxSize / 2.0){
            // NEGATIVE CHECK
            dq1[i] += boxSize;
        } // END else if (dq1[i] < -boxSize[i] / 2.0)
        
    } // END for (i = 0; i < DIM; i++)
    
    return;
}

//
//  FUNCTION: collision
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//  Function updates the post-collision state and tangent vectors after a collision has occurred.
//
//  Reference: Dinius, J. and J. Lega. TBD
//
//  INPUTS:
//  i - index of first disk involved in collision
//  j - index of first disk involved in collision
//
//  OUTPUTS:
//  (none)
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  phaseDim
//  dq1
//  DIM
//  nDisks
//  beta
//  alpha
//  
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  y   (velocity components of state vector and position and momentum components of tangent vectors are updated 
//       after collision) 
//  yi  (dummy variable)
//  yj  (dummy variable)
//  v   (dummy variable)
//  vq  (updated dot product of relative momentum and relative position of disks undergoing collision)
//  vv  (updated dot product of relative momentum with itself of disks undergoing collision)
//  dq  (dummy variable)
//  dv  (dummy variable)
//  dqc (dummy variable)
//  dq1 (dummy variable, used to store shortest distance between disk centers)
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
void collision(int i, int j)
{
    int ii,jj; // loop indices
    int pInd;  // index into tangent vector components
    double dqq, dvq, dqcv, dtc, velOff; // temporary variables for tangent vector updates
    
    for (ii = 0; ii < DIM; ii++){
        yi[ii] = y[DIM*i+ii]; // position vector of disk i
        yj[ii] = y[DIM*j+ii]; // position vector of disk j
        v[ii]  = y[DIM*(nDisks+j)+ii] - y[DIM*(nDisks+i)+ii]; // relative momentum between disks i and j
    } // END for (ii = 0; ii < DIM; ii++)
    
    // FIND SHORTEST DISTANCE CONNECTING CENTERS-OF-MASS OF DISKS i AND j
    image(yj,yi);
    
    // INITIALIZE vq AND vv
    vq = 0.0;
    
    // COMPUTE vq EQUALS THE DOT PRODUCT OF RELATIVE POSITION VECTOR WITH RELATIVE MOMENTUM VECTOR AND vv EQUALS THE DOT PRODUCT OF THE RELATIVE MOMENTUM VECTOR WITH ITSELF
    for (ii = 0; ii < DIM; ii++){
        vq    += dq1[ii]*v[ii];
    } // END for (ii = 0; ii < DIM; ii++)
    
    // UPDATE MOMENTA OF DISKS i AND j AFTER COLLISION (SEE REFERENCES)
    for (ii = 0; ii < DIM; ii++){
        y[DIM*(nDisks+i)+ii] += vq * dq1[ii];
        y[DIM*(nDisks+j)+ii] -= vq * dq1[ii];
    } // END for (ii = 0; ii < DIM; ii++)
    
    // UPDATE TANGENT VECTORS ASSOCIATED WITH DISKS i AND j AFTER COLLISION (SEE REFERENCES)
    for (ii = 1; ii < nlya+1; ii++){
        // FOR ALL TANGENT VECTORS
        
        // DETERMINE STARTING INDEX OF CURRENT TANGENT VECTOR WITHIN y
        pInd = ii*phaseDim;
        
        // SET PRE-COLLISION PERTURBATIONS AND RELATIVE QUANTITIES
        for (jj = 0; jj < DIM; jj++){
            dq[jj] = y[pInd+DIM*j+jj] - y[pInd+DIM*i+jj];                   // relative position perturbation between disks i and j before collision
            dv[jj] = y[pInd+DIM*(nDisks+j)+jj] - y[pInd+DIM*(nDisks+i)+jj]; // relative momentum perturbation between disks i and j before collision
        } // END for (jj = 0; jj < DIM; jj++)
        
        // INITIALIZE DOT PRODUCT ACCUMULATORS
        dqq = 0.0;
        dvq = 0.0;
        
        for (jj = 0; jj < DIM; jj++){
            dqq += dq[jj]*dq1[jj]; // dqq EQUALS DOT PRODUCT OF RELATIVE POSITION PERTURBATION VECTOR WITH RELATIVE POSITION VECTOR
            dvq += dv[jj]*dq1[jj]; // dvq EQUALS DOT PRODUCT OF RELATIVE MOMENTUM PERTURBATION VECTOR WITH RELATIVE POSITION VECTOR
		} // END for (jj = 0; jj < DIM; jj++)
        
        // COMPUTE TIME OFFSET OF COLLISION IN PERTURBED TRAJECTORY (SEE REFERENCES)
        dtc = -dqq / vq;
        dqcv = 0.0;
        
        // COMPUTE DELTA RELATIVE POSITION OFFSET VECTOR BETWEEN COLLISIONS (SEE REFERENCES)
        for (jj = 0; jj < DIM; jj++){
            dqc[jj] = dq[jj] + v[jj]*dtc;
			dqcv += v[jj]*dqc[jj]; // dqcv EQUALS DOT PRODUCT OF DELTA RELATIVE POSITION OFFSET VECTOR WITH RELATIVE MOMENTUM VECTOR
        } // for (jj = 0; jj < DIM; jj++)
        
        // UPDATE TANGENT VECTOR COMPONENTS ASSOCIATED WITH DISKS i AND j AFTER COLLISION (SEE REFERENCES)
        for (jj = 0; jj < DIM; jj++){
            velOff = (dvq + dqcv)*dq1[jj] + vq*dqc[jj];
            // \delta q part
			y[pInd+DIM*i+jj] += vq*dq1[jj]*dqq/vq;
            y[pInd+DIM*j+jj] -= vq*dq1[jj]*dqq/vq;
            // \delta p part
            y[pInd+DIM*(nDisks+i)+jj] += velOff;
            y[pInd+DIM*(nDisks+j)+jj] -= velOff;
        } // END for (jj = 0; jj < DIM; jj++)
        
    } // END for (ii = 1; ii < nlya+1; ii++)
    
    return;
}

//
//  FUNCTION: checkOverlap
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//  Function checks if any pair of disks overlaps.
//
//  INPUTS:
//  (none)
//
//  OUTPUTS:
//  1 - if any pair of disks overlap
//  0 - if no pair of disks overlap
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  nDisks
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  (none)
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
int checkOverlap(void)
{
    int i,j; // loop indices
    for (i = 0; i < nDisks - 1; i++){
        // FOR ALL DISKS (EXCEPT THE LAST)
        for (j = i+1; j < nDisks; j++){
            // CHECK IF DISK i COLLIDES WITH ANY SUBSEQUENTLY INDEXED DISKS
            if (overlap(i, j)){
                // AN OVERLAP WAS DETECTED BETWEEN DISKS i AND j, RETURN VALUE INDICATING OVERLAP
                return 1;
            } // END if (overlap(i, j))
            
        } // END for (j = i+1; j < nDisks; j++)
        
    } // END for (i = 0; i < nDisks - 1; i++)
    
    // NO OVERLAP WAS DETECTED, RETURN VALUE INDICATING NO OVERLAP
    return 0;
}

//
//  FUNCTION: overlap
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//  Function checks if any single interacting disk pair overlaps.
//
//  INPUTS:
//  i - index of disk 1
//  j - index of disk 2
//
//  OUTPUTS:
//  1 - if disks i and j overlap
//  0 - if disks i and j don't overlap
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  DIM
//  y
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  dq1 (updated through call to image() )
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
int overlap(int i, int j)
{
    int ii; // loop index
    
    for (ii = 0; ii < DIM; ii++){
        yi[ii] = y[DIM*i+ii]; // position vector of disk i
        yj[ii] = y[DIM*j+ii]; // position vector of disk J
    } // END for (ii = 0; ii < DIM; ii++)
    
    // FIND SHORTEST DISTANCE BETWEEN DISKS i AND j
    image(yi,yj);
    
    // INITIALIZE ACCUMULATOR FOR DISTANCE BETWEEN DISK CENTERS OF i AND j
    double s = 0.0;
    
    // COMPUTE LENGTH (SQUARED) OF DISTANCE BETWEEN DISK CENTERS
    for (ii = 0; ii < DIM; ii++){
        s += pow(dq1[ii],2.0);
    } // END for (ii = 0; ii < DIM; ii++)
    
    // COMPARE COMPUTED LENGTH (SQUARED) WITH SQUARE OF DISK DIAMETER (ASSUMES DISK DIAMETER IS 1)
    if (s < 1.0) 
    {
        // IF DISTANCE IS LESS THAN DISK DIAMETER, DECLARE THAT OVERLAP HAS OCCURRED
        return 1;
    } // END if (s < 1.0)
    
    // IF DISTANCE BETWEEN DISK CENTERS IS GREATER THAN (OR EQUAL TO) 1, NO OVERLAP HAS OCCURRED
    return 0;
}

//
//  FUNCTION: initialize
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//  Function initializes simulation quantities.
//
//  INPUTS:
//  (none)
//
//  OUTPUTS:
//  (none)
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  boxSize
//  phaseDim
//  nlya
//  nDisks
//  DIM
//  bigTime
//  noColl
//  maxFlight
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  y           (state vector and orthonormal basis spanning tangent space are initialized)
//  cum         (initialize vector to all zeros )
//  collArray   (initialize with first collision times and partners for each disk)
//  tumbleArray (initialize with first tumble times for each disk)
//  stats       (initialize members of structure with all zeros) 
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
void initialize(void)
{
    int i,j; // loop indices
    int n;   // round-off value of square root of nDisks calculation (check if nDisks is a perfect square).  This is used for initial configuration of disk positions.
    //int sqrtN = sqrt(nDisks);        // square root of number of disks.  This is used for initial configuration of disk positions.  Some weirdness with VS10's cmath: integer type is not supported for sqrt function (?) fix is below
    int sqrtN = (int)(sqrt((double)(nDisks)));        // square root of number of disks.  This is used for initial configuration of disk positions.
    double offSet = 0.0001243547654; // small additive offset to ensure that initial positions are well-defined within the simulation box
    double t1; // temporary variable for storing collision times between disks.  This is used in the initialization of collision times and partners for each disk.
    
    // ALLOCATE MEMORY TO STORE THE (TWO-DIMENSIONAL) MEAN OF MOMENTA OF THE DISKS
    double *mean;
    mean = new double [DIM];
    
	// INITIALIZE CLV STORAGE COUNTER (SAVE MEMORY)
	countCLV = 0;

    // INITIALIZE ALL ELEMENTS OF y TO ZERO (THE NECESSARILY UPDATED STATES WILL BE UPDATED BELOW)
    for (i = 0; i < (nlya+1)*phaseDim; i++){
        y[i] = 0.0;
    } // END for (i = 0; i < (nlya+1)*phaseDim; i++)
    
    // INITIALIZE PARAMETER n USED FOR DETERMINING INITIAL DISK LOCATIONS
    if (sqrtN*sqrtN == nDisks){
        // IF nDisks IS A PERFECT SQUARE, SET n EQUAL TO sqrtN
        n = sqrtN;
    } // END if (sqrtN*sqrtN == nDisks)
    
    else {
        // OTHERWISE, SET n TO THE INTEGER AFTER sqrtN
        n = sqrtN + 1;
    } // END else

	// SET DISK SPACING WIDTH IN X (dx) AND Y (dy)
	double dx = boxSize / ((double) n);
	double dy = dx; // square box

	for (i = 0; i < nDisks; i++){

		// SET Y-COMPONENT OF DISK i 
		y[DIM*i+1] = dy / 2.0 + (i / n) * dy + offSet;

		// SET X-COMPONENT OF DISK i
		if ( (i / n) % 2 == 0){
			// FOR EVERY OTHER ROW, STAGGER THE DISKS BY dx / 2 (CRYSTAL LATTICE PACKING)
			y[DIM*i] = (i % n) * dx + dx / 4.0 + offSet;
		} // END if ( (i / n) % 2 == 0)
		else {
			y[DIM*i] = dx / 2.0 + (i % n) * dx + dx / 4.0 + offSet;
		} // END else

	} // END for (i = 0; i < nDisks; i++)
    // ENFORCE PERIODIC BOUNDARY CONDITIONS
    boxSet();
    
    // INITIALIZE ACCUMULATOR OF MOMENTUM AVERAGE IN EACH DIRECTION 
    for (i = 0; i < DIM; i++){
        mean[i] = 0.0;
    }
    
	// initialize randome number generator with desired seed
	default_random_engine generator( (unsigned long)iSeed );
	normal_distribution<double> distribution(0.0,1.0);
		
    for (i = 0; i < nDisks; i++){
        
        for (j = 0; j < DIM; j++){
            // DRAW INITIAL MOMENTA FOR EACH DISK FROM A NORMAL DISTRIBUTION OF MEAN 0 AND STANDARD DEVIATION EQUAL TO SQRT(TEMPERATURE). TEMPERATURE IS HELD CONSTANT (AT 1) THROUGHOUT THE SIMULATION.
            y[DIM*(nDisks+i)+j] = distribution(generator);
            // INCREMENT THE COMPUTATION OF THE MEAN MOMENTUM
            mean[j] += y[DIM*(nDisks+i)+j] / nDisks;
        } // END for (j = 0; j < DIM; j++)
        
    } // END for (i = 0; i < nDisks; i++)
    
    // TRANSLATE EACH DISK'S MOMENTUM BY -MEAN SO THAT THE TOTAL SYSTEM CENTER-OF-MASS HAS ZERO MOMENTUM
    for (int i = 0; i < nDisks; i++){
        for (int j = 0; j < DIM; j++){
            y[DIM*(nDisks+i)+j] -= mean[j];
        } // END for (int j = 0; j < DIM; j++)
    
    } // END for (int i = 0; i < nDisks; i++)
    
    // INITIALIZE THE ACCUMULATOR FOR KINETIC ENERGY
    double ke = 0.0;
    
    // COMPUTE THE KINETIC ENERGY
    for (i = 0; i< nDisks; i++){
        for (j = 0; j < DIM; j++){
            ke += 0.5*pow(y[DIM*(nDisks+i)+j],2.0);
        } // END for (j = 0; j < DIM; j++)
        
    }  // END for (i = 0; i< nDisks; i++)
    
    // RESCALE EACH DISK VELOCITY SO THAT THE TOTAL KINETIC ENERGY IS EQUAL TO nDisks
    double c1 = sqrt(nDisks / ke); // kinetic energy scaling factor
    
    for (i = 0; i < nDisks; i++){
        
        for (int j = 0; j < DIM; j++){
            y[DIM*(nDisks+i)+j] *= c1;
        } // END for (int j = 0; j < DIM; j++)
        
    } // END for (i = 0; i < nDisks; i++)

    // INITIALIZE ORTHONORMAL TANGENT VECTORS TO THE STANDARD BASIS IN R^{4*nDisks} AND INITIALIZE THE ACCUMULATOR OF THE LOG OF THE TANGENT VECTOR STRETCHING FACTORS
    for (i = 1; i < nlya+1; i++){
        y[i*(phaseDim + 1) - 1] = 1.0;
        cum[i-1] = 0.0;
    } // END for (i = 1; i < nlya+1; i++)
    
    // INITIALIZE COLLISION EVENT STRUCTURE WITH LARGE TIME AND NULL PARTNER (noColl)
    for (i = 0; i < nDisks; i++){
        collArray[i].time    = bigTime;
        collArray[i].index   = i;
        collArray[i].partner = noColl;
    } // END for (i = 0; i < nDisks; i++)
    
    // TO INITIALIZE COLLISION TIMES, FOLLOW THE SAME PROCESS AS IN updateTimes ROUTINE:
    for (i = 0; i < nDisks - 1; i++){
        // SEARCH FOR THE MINIMUM TIME AND THE ASSOCIATED PARTNER OF THE NEXT COLLISION FOR DISK i
        for (j = i+1; j < nDisks; j++){
            // SEARCH OVER ALL DISKS WITH INDEX GREATER THAN i
            
            // COMPUTE TIME OF COLLISION BETWEEN DISKS i AND j
            t1 = binTime(i, j);
            
            if (t1 > 0.0){
                
                // IF COLLISION TIME IS POSITIVE, THEN THE COLLISION IS A VALID ONE
                if (t1 < collArray[i].time){
                    // IF THE COLLISION TIME BETWEEN DISKS j AND i IS SMALLER THAN THE CURRENT CALCULATED COLLISION 
                    // TIME FOR DISK i, UPDATE THE TIME AND PARTNER FOR DISK i
                    collArray[i].time    = t1;
                    collArray[i].partner = j;
                } // END if (t1 < collArray[i].time)
                
                if (t1 < collArray[j].time){
                    // IF THE COLLISION TIME BETWEEN DISKS j AND i IS SMALLER THAN THE CURRENT CALCULATED COLLISION 
                    // TIME FOR DISK j, UPDATE THE TIME AND PARTNER FOR DISK j
                    collArray[j].time    = t1;
                    collArray[j].partner = i;
                } // END if (t1 < collArray[j].time)
                
            } // END if (t1 > 0.0)
            
        } // END for (j = i+1; j < nDisks; j++)
        
    } // for (i = 0; i < nDisks - 1; i++)
    
    for (i = 0; i < nDisks; i++){
        
        // CHECK FOR MAX FLIGHT CONDITION
        for (j = 0; j < DIM; j++){
            
            if ( fabs(collArray[i].time * y[DIM*(i+nDisks)+j]) > maxFlightDblArray[j] ){
                // IF THE DISTANCE TRAVELED IN EITHER SPATIAL DIRECTION BEFORE A VALID COLLISION INVOLVING DISK i EXCEEDS THE MAXIMUM FLIGHT CONDITION maxFlight, THEN SET THE COLLISION TIME AND PARTNER FOR DISK i ACCORDINGLY
                collArray[i].time    = fabs(maxFlightDblArray[j] / y[DIM*(i+nDisks)+j]);
                collArray[i].partner = maxFlight;
            } // END if ( fabs(collArray[i].time * y[DIM*(i+nDisks)+j]) > maxFlightDblArray[j] )
            
        } // END for (j = 0; j < DIM; j++)
        
    } // END for (i = 0; i < nDisks; i++)
    
    // CHECK TO MAKE SURE THAT NO DISKS OVERLAP
    if (checkOverlap()){
        // EXIT THE SIMULATION IF DISKS OVERLAP
        cerr << "Disks shouldn't overlap" << endl;
        exit(1);
    } // END for (i = 0; i < nDisks; i++)
    
    // PERFORM CLEANUP OF LOCAL MEMORY ALLOCATED FOR THE COMPUTATION OF mean
    delete [] mean;
    
    return;
}

//
//  FUNCTION: mgsr
//  MODULE:   diskDynamics
//
//  DESCRIPTION: 
//  Function performs modified Gram-Schmidt reorthonormalization to prevent collapse of all tangent vectors onto the direction of maximal instability in forward-time.
//
//  Reference:
//  Trefethen, L. and D. Bau III.  Numerical Linear Algebra, SIAM 1997. Pgs. 56-68
//
//  INPUTS:
//  (none)
//
//  OUTPUTS:
//  (none)
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  Time
//  phaseDim
//  nlya
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  norm (the set of stretching factors of the tangent vectors is updated)
//  qi   (dummy variable)
//  y    (the set of tangent vectors are orthonormalized)
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
void mgsr(void)
{
    int i,j,k; // loop indices
    double riisq, rij, tmp1, rii; // local variables for orthogonalization/normalization
    vector<double> Rmat, Gmat, invR, ps;
	// Rmat is written out by row; Gmat by column

	// IF COMPUTING CLVs, WRITE OUT CURRENT PHASE SPACE TRAJECTORY POINT
	if (Time > clvSettleTime && ((countCLV % modCLV) ==0)){
		for (i=0;i<phaseDim;i++){
			ps.push_back(y[i]);
		}
		traj.push_back(ps);
	}
    for (i = 1; i < nlya+1; i++){
        
        // COMPUTE rii IN ALGORITHM 8.1 FROM REFERENCE
        // rii IS THE NORM OF THE COLUMN VECTOR IN y ASSOCIATED WITH INDEX i 
        riisq = 0.0;
        for (j = 0; j < phaseDim; j++){
            riisq += pow(y[i*phaseDim+j],2.0);
        } // END for (j = 0; j < phaseDim; j++)
        

        rii = sqrt(riisq);

		if (Time > clvSettleTime && ((countCLV % modCLV) == 0) && lastPass){
			// PUT IN 0'S IN LOWER TRIANGULAR BLOCK (EXCEPT FOR DIAGONAL)
			for (j = 0; j < i-1; j++){
				Rmat.push_back(0.);
			}
			// PUT DIAGONAL ELEMENT INTO R ARRAY (FOR CLV COMPUTATION)
			Rmat.push_back(rii);
		}

        // COMPUTE qi IN ALGORITHM 8.1 FROM REFERENCE
        // qi IS THE NORMALIZATION OF COLUMN VECTOR IN y ASSOCIATED WITH INDEX i
        for (j = 0; j < phaseDim; j++){
            qi[j] = y[i*phaseDim+j] / rii;
        } // END for (j = 0; j < phaseDim; j++)

        for (j = i+1; j < nlya + 1; j++){
            // COMPUTE rij IN ALGORITHM 8.1 FROM REFERENCE
            // rij IS THE DOT PRODUCT OF COLUMN VECTOR IN y ASSOCIATED WITH INDEX j AND qi (PROJECTION)
            rij = 0.0;
            for (k = 0; k < phaseDim; k++){
                rij += qi[k]*y[j*phaseDim+k];
            } // END for (k = 0; k < phaseDim; k++)
            
			if (Time > clvSettleTime && ((countCLV % modCLV) == 0) && lastPass ){
				// PUT DIAGONAL ELEMENT INTO R ARRAY (FOR CLV COMPUTATION)
				Rmat.push_back(rij);
			}
            // FROM THE THE COLUMN VECTOR IN y ASSOCIATED WITH INDEX j, REMOVE THE PROJECTION OF qi ONTO THE COLUMN VECTOR IN y ASSOCIATED WITH INDEX j (THIS IS THE ORTHOGONALIZATION STEP)
            for (k = 0; k < phaseDim; k++){
                y[j*phaseDim+k] -= rij*qi[k];
            } // END for (k = 0; k < phaseDim; k++)
            
        } // END for (j = i+1; j < nlya + 1; j++)
        
    } // END for (i = 1; i < nlya; i++)
    
    // NORMALIZE ALL VECTORS AND RECORD THE STRETCHING/CONTRACTION FACTORS FOR LYAPUNOV EXPONENTS CALCULATION
    for (i = 1; i < nlya+1; i++){
        tmp1 = 0.0;
        for (j = 0; j < phaseDim; j++){
            tmp1 += pow(y[i*phaseDim+j],2.0);   
        } // END for (j = 0; j < phaseDim; j++)
        
        norm[i-1] = sqrt(tmp1);
        
        for (j = 0; j < phaseDim; j++){
            y[i*phaseDim+j] /= norm[i-1];
			if (Time > clvSettleTime && ((countCLV % modCLV) == 0)){
				// STORE GS BASIS (FOR CLV COMPUTATION)
				Gmat.push_back( y[i*phaseDim+j] );
			}
        } // END for (j = 0; j < phaseDim; j++)
        
    } // END for (i = 1; i < nlya+1; i++)
    
	if (Time > clvSettleTime && ((countCLV % modCLV) == 0)){
		if (lastPass){
			// INVERT THE R MATRIX
			invR = invertRmat( Rmat );
			Rinv.push_back(invR);
		}
		// PUSH VECTORS INTO VECTOR CONTAINER
		GS.push_back(Gmat);
	}

	countCLV += 1;

	Gmat.clear();
	invR.clear();
	Rmat.clear();
	ps.clear();

    return;
}

//
//  FUNCTION: update
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//  Function updates accumulation of the log of the tangent vector stretching factors and the time average of the accumulation (Lyapunov exponents).  The option exists to record batch averages of the accumulated logarithms of the stretching factors; turn this functionality on by setting computeBatchAvg to a nonzero value.
//
//  INPUTS:
//  (none)
//
//  OUTPUTS:
//  (none)
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  nlya
//  Time
//  norm
//  computeBatchAvg
//  numMbatch
//  batchData
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//  cum         (the log of the current stretching factors is added to the previous values)
//  lSpec       (the time average of cum is updated)
//  cumBatch    (the log of the current stretching factors is added to the previous values, or reset to zero if the 
//               number of batch samples is equal to numMbatch)
//  avgCumBatch (the average of cumBatch is recorded when the number of batch samples equals numMbatch)
//  countMbatch (iterated or reset to zero if counter limit has been reached)
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            10/08/11
//  Dinius, J.       Comments added for v1.0 release    05/27/13
//
void update(void)
{
    int i; // loop index
    for (i = 0; i < nlya; i++){
        // FOR EACH TANGENT VECTOR:
        
        // ACCUMULATE THE LOG OF THE STRETCHING FACTOR OF THE iTH TANGENT VECTOR
        cum[i] += log(norm[i]);
        
        // TAKE THE TIME AVERAGE OF cum; THIS IS THE CURRENT ESTIMATE OF THE iTH LYAPUNOV EXPONENT
        lSpec[i] = cum[i] / Time;
        
    } // for (i = 0; i < nlya; i++)
    
    return;
}

//
//  FUNCTION: invertRmat
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//
//  Reference:
//  
//  INPUTS:
//  Rvec- R matrix from MGSR (in vector form)
//
//  OUTPUTS:
//  invRred- inverted R matrix (with zeros in lower-triangular block trimmed to save memory)
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//  phaseDim
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            11/02/13
//
vector<double> invertRmat(vector<double> Rvec)
{
	int i, j, k;
	vector<double> invR,invRred;
	//int phaseDim = 4; //unit test
	
	// INITIALIZE INVERSE CONTAINER
	invR.assign(phaseDim*phaseDim,0.);

	for (i=0; i<phaseDim; i++){
		invR.at(phaseDim*i+i) = 1. / Rvec.at(phaseDim*i+i);
		
		for (j=0; j<i; j++){
			for (k=0; k<i; k++){
				invR.at(phaseDim*j+i) += invR.at(phaseDim*j+k)*Rvec.at(phaseDim*k+i);
			}
		}

		for (j=0; j<i; j++){
			invR.at(phaseDim*j+i) /= -Rvec.at(phaseDim*i+i);
		}
	}

	// TRIM ZEROS (SAVE ON STORAGE BURDEN)
	for (i=0; i<phaseDim; i++){
		for (j=i; j<phaseDim; j++){
			invRred.push_back( invR.at(i*phaseDim+j) );
		}
	}

	invR.clear();

	return invRred;
}

//
//  FUNCTION: computeCLVs
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//
//  Reference:
//  
//  INPUTS:
//
//  OUTPUTS:
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            11/02/13
//
void computeCLVs()
{
	int i, j, k, counter, fcount;
	vector<double> C,Cp,V,Vp,invRfull,invRred,G,Gp,expBack, psFinal, GSfinal;
	double nrm;
	vector<double> cumback;
	vector<double> ps;
	vector< vector<double> > GSloc, trajLoc;
	double dt = (double)(modCLV) * stepSize * (double)(nOrtho);
	double t  = 0.;
	double tprop = 0;
	double dtInner = stepSize * (double)(nOrtho);
	fcount = 0;
	double currTime = 0.;
	
	int timesteps = modCLV*nOrtho;
	
	vector<double> tvecloc;
	
	// RESET modCLV (WE WANT EACH ORTHONORMALIZATION STORED FROM NOW ON)
	modCLV = 1;

	lastPass = 1;

	// CREATE LOCAL COPIES OF GS, Rinv, AND traj VARIABLES (CLEAR GLOBALS FOR IN-BETWEEN UPDATES)
	GSloc   = GS;
	// RinvLoc = Rinv; not needed
	trajLoc = traj;

	GS.clear();
	traj.clear();

	// INITIALIZE BACKWARD ACCUMULATOR (FOR LYAPUNOV EXPONENT VERIFICATION)
	cumback.assign(phaseDim,0.);
	expBack.assign(phaseDim,0.);

	// INITIALIZE C TO THE IDENTITY MATRIX
	C.assign(phaseDim*phaseDim,0.);
	for (i=0; i<phaseDim; i++){
		C.at(i*phaseDim+i) = 1.;
	}

	// INITIALIZE V TO THE LAST GS STEP 
	// ( V(end) = G(end)*C(end) = G(end)*I = G(end) )
	V = GSloc.back();

	// CLEAN UP MEMORY ON THE FLY (FINAL TIME POINT NOT NEEDED ANY LONGER)
	psFinal = trajLoc.back();
	trajLoc.pop_back();
	GSloc.pop_back();

	//for (i=GSloc.size()-1; i>0; i--){
	int initGSsize = GSloc.size();
	i = initGSsize-1;

	int countAcc = 0;
	int counterAcc = 0;
	int nrmCntr;

	while (!GSloc.empty() || !trajLoc.empty()){
		i -= 1;
		cout << i << " " << expBack.at(0) << endl;

		// WRITE OUT CURRENT TIME (t), COVARIANT VECTORS (V) AND EXPONENTS (expBack)
		clvData.write((char *) &t, sizeof(double));
		for (j=0; j<phaseDim; j++){
			for (k=0; k<phaseDim; k++){
				clvData.write((char *) &V.at(j*phaseDim+k), sizeof(double));
			}
		}
		for (j=0; j<phaseDim; j++){
			clvData.write((char *) &expBack.at(j), sizeof(double));
		}

		// POPULATE INITIAL CONDITION (OVERWRITE y ARRAY SINCE IT'S NO LONGER IN USE)
		ps = trajLoc.back(); // y(n-1)
		G  = GSloc.back();  // G(n-1)

		// DELETE LAST POINT IN VECTORS ps AND G (SAVE MEMORY)
		trajLoc.pop_back();
		GSloc.pop_back();

		for (j=0; j<phaseDim; j++){
			for (k=0; k <= phaseDim; k++){
				if (k==0){
					y[j] = ps.at(j);
				}
				else{
					y[k*phaseDim + j] = G.at( (k-1)*phaseDim+j );
				}
			}
		}
		
		// INITIALIZE COLLISION TIMES
		initializeCLVcollTimes();

		// CLEAR LOCALS
		G.clear();
		ps.clear();

		// PROPAGATE INTERMEDIATE STATES (FROM i-1 TO i) TO SAVE ON MEMORY ALLOCATION
		nrmCntr = 0;
		while (fcount <= timesteps){
			if ( (fcount % nOrtho) == 0 ){
				// PERFORM GRAM-SCHMIDT REORTHONORMALIZATION OF TANGENT VECTORS
				tvecloc.push_back(tprop);
				mgsr();
				nrmCntr+=1;
			}
			hardStep(stepSize);
			tprop += stepSize;
			if (checkOverlap()){
				// IF DISKS OVERLAP, EXIT
				cout << "Disks overlap... exiting" << endl;
				//exit(1);
				//cout << "Disks overlap @ time " << Time << endl;
			} // END if (checkOverlap())

			fcount+= 1;
		}

		// REMOVE LAST ELEMENT OF GS (NOT NEEDED)
		GS.pop_back();

		while (!GS.empty()){
		//for (jj=GS.size()-1; jj>0; jj--){
			// UNPACK RED RINV MATRIX  (Rred^{-1}(i))
			invRred = Rinv.back(); //
			
			// DELETE LAST ENDPOINT IN Rinv (SAVE MEMORY)
			Rinv.pop_back();

			counter = 0;
			invRfull.assign(phaseDim*phaseDim,0.);

			// PUT ZEROS BACK INTO R^{-1}(i)
			for (j=0;j < phaseDim; j++){ //move debug file stuff here.
				for (k=j;k < phaseDim; k++){
					invRfull.at(j*phaseDim+k) = invRred.at(counter);
					counter += 1;
				}
			}

			// C(i-1) = R^{-1}(i)*C(i-1)
			Cp = matrixMultiply( rowToColumn(invRfull) , C);
			
			// RESCALE THE COLUMNS BY THEIR RESPECTIVE NORMS (REMOVE STRETCHING AND CONTRACTION FACTORS)
			for (j=0; j < phaseDim; j++){
				nrm = columnNorm(Cp,j);
				cumback.at(j) += log(nrm);
				Cp  = rescaleCol(Cp,j,nrm);

				// LYAPUNOV EXPONENTS COMPUTATION
				expBack.at(j) = cumback.at(j) / (t+currTime);
			}

			// INITIALIZE G(i-1)
			Gp = GS.back();
			
			// V(i-1) = G(i-1)*C(i-1)
			Vp = matrixMultiply(Gp,Cp);

			// UPDATE VALUES OF PLACEHOLDERS C,V FOR NEXT LOOP THROUGH
			C = Cp;
			
			currTime += dtInner;

			Cp.clear();
			Gp.clear();
			GS.pop_back();
		} // while (!GS.empty())

		V = Vp;
		
		tprop = 0.;
		fcount = 0;
		currTime = 0.;

		// UPDATE TIME
		t += dt;

	} // while (!GSloc.empty() || !trajLoc.empty())

	clvData.close();

	V.clear();

	return;
}

//
//  FUNCTION: rowToColumn
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//
//  Reference:
//  
//  INPUTS:
//
//  OUTPUTS:
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            11/02/13
//
vector<double> rowToColumn( vector<double> V )
{
	vector<double> rowToColSwap;
	rowToColSwap.assign(phaseDim*phaseDim,0.);

	for (int i=0; i<phaseDim; i++){
		for (int j=0; j<phaseDim; j++){
			rowToColSwap.at(j*phaseDim+i) = V.at(i*phaseDim+j);
		}
	}

	return rowToColSwap;
}

//
//  FUNCTION: matrixMultiply
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//
//  Reference:
//  
//  INPUTS:
//
//  OUTPUTS:
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            11/02/13
//
vector<double> matrixMultiply(vector<double> v1, vector<double> v2)
{
	vector<double> prod;
	//int phaseDim=4; //DEBUG
	prod.assign(phaseDim*phaseDim,0.);

	for (unsigned i=0; i < phaseDim; i++){ //COLUMN
		for (unsigned k=0; k < phaseDim; k++){ //ROW
			for (unsigned j = 0; j<phaseDim; j++){
				prod.at(i*phaseDim+k) += v1.at(j*phaseDim+k)*v2.at(i*phaseDim+j); //pack by row
				//prod.at(k*phaseDim+i) += v1.at(i*phaseDim+j)*v2.at(j*phaseDim+k);
			}
		}
	}
	return prod;
}

//
//  FUNCTION: columnNorm
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//
//  Reference:
//  
//  INPUTS:
//
//  OUTPUTS:
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            11/02/13
//
double columnNorm(vector<double> v, int col)
{
	double sum = 0.;
	for (unsigned j = 0; j < phaseDim; j++){
		sum += v.at(col*phaseDim+j)*v.at(col*phaseDim+j);
	}
	return sqrt(sum);
}

//
//  FUNCTION: rescaleCol
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//
//  Reference:
//  
//  INPUTS:
//
//  OUTPUTS:
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            11/02/13
//
vector<double> rescaleCol(vector<double> v, int col, double nrm)
{
	vector<double> vn = v;
	for (unsigned j = 0; j < phaseDim; j++){
		vn.at(col*phaseDim+j) /= nrm;
	}
	
	return vn;
}

//
//  FUNCTION: initializeCLVcollTimes
//  MODULE:   diskDynamics
//
//  DESCRIPTION:
//
//  Reference:
//  
//  INPUTS:
//
//  OUTPUTS:
//
//  GLOBAL INPUTS (UNCHANGED DURING CALL):
//
//  GLOBAL OUTPUTS (UPDATED DURING CALL):
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            11/02/13
//
void initializeCLVcollTimes()
{
	int i,j; // loop indices
    double t1;
	// INITIALIZE COLLISION EVENT STRUCTURE WITH LARGE TIME AND NULL PARTNER (noColl)
    for (i = 0; i < nDisks; i++){
        collArray[i].time    = bigTime;
        collArray[i].index   = i;
        collArray[i].partner = noColl;
    } // END for (i = 0; i < nDisks; i++)
    
    // TO INITIALIZE COLLISION TIMES, FOLLOW THE SAME PROCESS AS IN updateTimes ROUTINE:
    for (i = 0; i < nDisks - 1; i++){
        // SEARCH FOR THE MINIMUM TIME AND THE ASSOCIATED PARTNER OF THE NEXT COLLISION FOR DISK i
        for (j = i+1; j < nDisks; j++){
            // SEARCH OVER ALL DISKS WITH INDEX GREATER THAN i
            
            // COMPUTE TIME OF COLLISION BETWEEN DISKS i AND j
            t1 = binTime(i, j);
            
            if (t1 > 0.0){
                
                // IF COLLISION TIME IS POSITIVE, THEN THE COLLISION IS A VALID ONE
                if (t1 < collArray[i].time){
                    // IF THE COLLISION TIME BETWEEN DISKS j AND i IS SMALLER THAN THE CURRENT CALCULATED COLLISION 
                    // TIME FOR DISK i, UPDATE THE TIME AND PARTNER FOR DISK i
                    collArray[i].time    = t1;
                    collArray[i].partner = j;
                } // END if (t1 < collArray[i].time)
                
                if (t1 < collArray[j].time){
                    // IF THE COLLISION TIME BETWEEN DISKS j AND i IS SMALLER THAN THE CURRENT CALCULATED COLLISION 
                    // TIME FOR DISK j, UPDATE THE TIME AND PARTNER FOR DISK j
                    collArray[j].time    = t1;
                    collArray[j].partner = i;
                } // END if (t1 < collArray[j].time)
                
            } // END if (t1 > 0.0)
            
        } // END for (j = i+1; j < nDisks; j++)
        
    } // for (i = 0; i < nDisks - 1; i++)

	for (i = 0; i < nDisks; i++){
	// CHECK FOR MAX FLIGHT CONDITION
        for (j = 0; j < DIM; j++){
            
            if ( fabs(collArray[i].time * y[DIM*(i+nDisks)+j]) > maxFlightDblArray[j] ){
                // IF THE DISTANCE TRAVELED IN EITHER SPATIAL DIRECTION BEFORE A VALID COLLISION INVOLVING DISK i EXCEEDS THE MAXIMUM FLIGHT CONDITION maxFlight, THEN SET THE COLLISION TIME AND PARTNER FOR DISK i ACCORDINGLY
                collArray[i].time    = fabs(maxFlightDblArray[j] / y[DIM*(i+nDisks)+j]);
                collArray[i].partner = maxFlight;
            } // END if ( fabs(collArray[i].time * y[DIM*(i+nDisks)+j]) > maxFlightDblArray[j] )
            
        } // END for (j = 0; j < DIM; j++)
	}

	return;
}
