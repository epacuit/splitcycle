'''
	*File: 	  domain_restriction.py
	*Author:	Nicholas Mattei (nicholas.mattei@nicta.com.au)
	*Date:	  March 18, 2014
  *
  * Copyright (c) 2014, Nicholas Mattei and NICTA
  * All rights reserved.
  *
  * Developed by: Nicholas Mattei
  *               NICTA
  *               http://www.nickmattei.net
  *               http://www.preflib.org
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted provided that the following conditions are met:
  *     * Redistributions of source code must retain the above copyright
  *       notice, this list of conditions and the following disclaimer.
  *     * Redistributions in binary form must reproduce the above copyright
  *       notice, this list of conditions and the following disclaimer in the
  *       documentation and/or other materials provided with the distribution.
  *     * Neither the name of NICTA nor the
  *       names of its contributors may be used to endorse or promote products
  *       derived from this software without specific prior written permission.
  *
  * THIS SOFTWARE IS PROVIDED BY NICTA ''AS IS'' AND ANY
  * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  * DISCLAIMED. IN NO EVENT SHALL NICTA BE LIABLE FOR ANY
  * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.	
	

About
--------------------
	This file tests a profile for being single-peaked.
'''

import sys
import copy
import glob

from preflibtools import io
from preflibtools import generate_profiles

# Implement of the Single Peaked Consistancy Algorithm detailed in
# B. Escoffier, J. Lang, and M. Ozturk, "Single-peaked consistency and its complexity".
# 2008 European Conference on Artificial Intelligence.
#
# Intuitivily, this algortihm finds an axis that is single peaked with respect to 
# the rmaps that are passed in, or it returns an empty axis (vector) of the candidates.
# This is achieved in time O(|rmaps|*|candmap|).
#
# Note that this algorithm only works for STRICT preferences.  If a non-strict 
# set of rankmaps is passed in, an error is returned.
def is_single_peaked(rmaps, candmap):
  for current in rmaps:
      if len(current) != len(candmap):
        print("is_single_peaked called with non-strict preferences")
        exit()

  orders = order_vectors(rmaps)
  fullorders = order_vectors(rmaps)

  #Build the order...
  leftside = []
  rightside = []
  last_cands = last_set(orders)
  
  # Only one last makes no constraints so iterate...
  while (len(last_cands) == 1):
    if len(leftside) < len(rightside):
      leftside.append(last_cands[0])
    else:
      rightside.insert(0,last_cands[0])
    orders = remove_cands(orders, last_cands)
    last_cands = last_set(orders)
  #Only break if we have != 1 last candidate, wither we quit, or we put on one each end.
  if len(last_cands) > 2:
    return []
  else:
    leftside.append(last_cands[0])
    rightside.insert(0,last_cands[1])
    orders = remove_cands(orders, last_cands)

  # While there are still unplaced candidates (not removed from every vote))
  while len(orders[0]) > 0:
    last_cands = last_set(orders)
    # Should never have more than 2...
    if len(last_cands) > 2:
      return []
    else:
      x_i = leftside[len(leftside)-1]
      x_j = rightside[0]

      #Check Conditions outlined by Lang.
      #If L={X}, Case 3
      if len(last_cands) == 1:
        x = last_cands[0]
        # if x_i < x < x_j => leftside + x
        if any(o.index(x_j) < o.index(x) and o.index(x) < o.index(x_i) for o in fullorders):
          leftside.append(x)
        # if x_j < x < x_i => x + right
        elif any(o.index(x) < o.index(x_j) and o.index(x_i) < o.index(x) for o in fullorders):
          rightside.insert(0, x)
        # Otherwise it doest nmatter and we put it either place...
        else:
          if len(leftside) < len(rightside):
            leftside.append(x)
          else:
            rightside.insert(0,x)
        # Restrict...
        orders = remove_cands(orders, last_cands)           

        #if L = {x, y}, Case 2c and 2d
        # if x_i < x < x_j < y ==> left+x and y+right
        # if x_j < x < x_i < y ==> left+y and x+right
        # if both, then contradiction...
        # if x_i < x < y < x_j ==> then this must be axis...
        # if x_j < y < x < x_i ==> then this must be axis...
      elif len(last_cands) == 2:
        C1 = False
        C2 = False
        x = last_cands[0]
        y = last_cands[1]
        x_i = leftside[len(leftside)-1]
        x_j = rightside[0]

        # Iterate over each of the orders and check for the C1 or C2 conditions or D1 or D2... Switch on these..
        for o in fullorders:
          #Condition D1:
          if o.index(x_i) > o.index(x) and o.index(x) > o.index(y) and o.index(y) > o.index(x_j):
            # The axis is the current voter restricted to the remainder
            temp_order = copy.copy(o)
            temp_order = remove_cands([temp_order], list(set(leftside + rightside)))[0]
            temp_order.reverse() ## Note that this reversed is the "increasing order of voter j"
            social_axis = leftside + temp_order + rightside
            if verify_orders_single_peaked_axis_strict(social_axis, fullorders): 
              return social_axis
            else:
              return []

          #Condition D2:
          if o.index(x_j) > o.index(y) and o.index(y) > o.index(x) and o.index(x) > o.index(x_i):
            # The axis is the current voter restricted to the remainder
            temp_order = copy.copy(o)
            temp_order = remove_cands([temp_order], list(set(leftside + rightside)))[0]
            social_axis = leftside + temp_order + rightside
            if verify_orders_single_peaked_axis_strict(social_axis, fullorders): 
              return social_axis
            else:
              return []

          #Condition C1:
          if o.index(x_i) > o.index(x) and o.index(x) > o.index(x_j) and o.index(x_j) > o.index(y):
            C1 = True
          #Condition C2:
          if o.index(x_j) > o.index(x) and o.index(x) > o.index(x_i) and o.index(x_i) > o.index(y):
            C2 = True

          # Short Circuit if we have C1 and C2 at any point...
          if C1 and C2:
            return []

        # Processing C1 or C2 if necessary:
        if C1:
          leftside.append(x)
          rightside.insert(0,y)
        else: # Do C2 or it doesn't matter...
          leftside.append(y)
          rightside.insert(0,x)
        orders = remove_cands(orders, last_cands)

  #Leftside + Rightside must be the social axis
  social_axis = leftside+rightside
  if verify_orders_single_peaked_axis_strict(social_axis, fullorders):
    return social_axis
  else:
    return []

# Helper function to find last place candidates
def last_set(orders):
  if len(orders) > 0 and len(orders[0]) > 0:
    # Make and return the set of last place candidates
    last_cands = set()
    for i in orders:
      last_cands.add(i[len(i)-1])
  
  return(list(last_cands))

# Helper function to compute the result of removing (set) of candidates from a list of orders.
def remove_cands(orders, cands_to_remove):
  projection = []
  for c_vote in orders:
    tvote = copy.copy(c_vote)
    for c_remove in cands_to_remove:
      tvote.remove(c_remove)
    projection.append(tvote)
  return projection

# Helper Function: Given cands --> rank, return a vector of unique vectors in the profile  
# that are just the orders of the candidates with index 0 == most prefered.
def order_vectors(rmaps):
  orders = []
  rank_to_candidate = io.rankmap_convert_rank_to_candidate(rmaps)
  for c_map in rank_to_candidate:
    c_vote = []
    for i in sorted(c_map.keys()):
      c_vote.append(c_map[i])
    orders.append(c_vote)
  return orders

# Verify that a profile of strict orders is single peaked w.r.t. the passed axis
def verify_orders_single_peaked_axis_strict(axis, orders):
  # print("Candidate Axis: " + str(axis))
  # print("Orders: " + str(orders))
  if len(orders) < 1 or len(axis) != len(orders[0]):
    return False

  temporders = copy.copy(orders)
  for corder in orders:
    #Peal off the top element
    split = axis.index(corder[0])
    # Reverse the left side and compare element by element on the restricted set.
    left = axis[:split]
    left.reverse()
    right = axis[split:]
    
    # print("Checking Left Side")
    restricted = remove_cands([corder], list(set(axis) - set(left)))
    restricted = restricted[0]
    #items should match element for element...
    if len(left) > 0 and not all(restricted[i] == left[i] for i in range(len(left))):
        print("Axis is not compatiable with order: " + str(corder))
        return False
    # print("Checking Right Side")
    restricted = remove_cands([corder], list(set(axis) - set(right)))
    restricted = restricted[0]
    #items should match element for element...
    if not all(restricted[i] == right[i] for i in range(len(right))):
        print("Axis is not compatiable with order: " + str(corder))
        return False

  return True


# Generate a random instance and test it for SP -- Output the axis if it is...
if __name__ == '__main__':

  ncand = 3
  nvoters = 100
  candmap = generate_profiles.gen_cand_map(ncand)
  #rmaps, rmapscounts = generate_profiles.gen_impartial_culture_strict(nvoters, cmap)
  rankmaps, rankmapcounts = generate_profiles.gen_single_peaked_impartial_culture_strict(nvoters, candmap)

  io.pp_profile_toscreen(candmap, rankmaps, rankmapcounts)

  social_axis = is_single_peaked(rankmaps, candmap)
  if social_axis != []:
    print("Single Peaked w.r.t " + str(social_axis))
  else:
    print("Not Single Peaked")


  # Test all the SOC's... for fun....
  files = glob.glob("./soc/*.soc")
  total = 0
  totalSP = 0
  for cfile in sorted(files):
    print("Testing: " + str(cfile))
    inf = open(cfile, "r")
    candmap, rankmaps, rankmapcounts, numvoters = io.read_election_file(inf)
    total += 1
    social_axis = is_single_peaked(rankmaps, candmap)
    if social_axis != []:
      print("Single Peaked w.r.t " + str(social_axis))
      totalSP += 1
    else:
      print("Not Single Peaked")
    inf.close()
  print("Parsed " + str(total) + " SOC files")
  print("Exactly " + str(totalSP) + " were single peaked")



  