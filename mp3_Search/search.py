# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP3. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
# Note that if you want to test one of your search methods, please make sure to return a blank list
#  for the other search methods otherwise the grader will not crash.
class MST:
    # def __init__(self, objectives, len_dist, all_ends):
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}
        # self.len_dist = len_dist
        # self.all_ends = all_ends

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
               # (i, j): self.distance_cmp(i, j, self.len_dist, self.all_ends)
               (i, j): self.mht(i, j)
                for i, j in self.cross(objectives)
            }

    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root

    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a)
        rb = self.resolve(b)
        if ra == rb:
            return False
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

    def mht(self, x, y):
        return abs(x[0]-y[0]) + abs(x[1]-y[1])

    ''' def distance_cmp(self, x, y, len_dist, all_ends):
        idx_x = all_ends.index(x)
        idx_y = all_ends.index(y)
        if (idx_x < idx_y):
            pair = (idx_x, idx_y)
        else:
            pair = (idx_y, idx_x)
        dist_xy = len_dist[pair]
        return dist_xy
        '''

import collections

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.start
    end = maze.waypoints[0]
    search_q = collections.deque();
    seen = []
    path = []
    prev_dict = {}
    rows    = maze.size.y
    columns = maze.size.x
    indices = maze.indices()
    
    search_q.append(start)
    cur = search_q.popleft()
    seen.append(cur)

    while cur != end:
        cur_neigh = []
        cur_x = cur[1]
        cur_y = cur[0]
        if(cur_y > 1):
            cur_neigh.append((cur_y-1,cur_x))
        if(cur_y < rows-2):
            cur_neigh.append((cur_y+1,cur_x))
        if(cur_x > 1):
            cur_neigh.append((cur_y,cur_x-1))
        if(cur_x < columns-2):
            cur_neigh.append((cur_y,cur_x+1))

        for i in range(len(cur_neigh)):
            checknavi_x = cur_neigh[i][1]
            checknavi_y = cur_neigh[i][0]
            if(maze.navigable(checknavi_y,checknavi_x)):
               if ((cur_neigh[i] not in seen) & (cur_neigh[i] not in search_q)):
                   search_q.append(cur_neigh[i])
                   cur_dict = cur_neigh[i]
                   prev_dict[cur_dict]=cur

        cur = search_q.popleft()
        seen.append(cur)

    path.append(end);
    cur = end
    seen.remove(cur)
    cur_x = end[1]
    cur_y = end[0]

    while (cur != start):
        cur_x = cur[1]
        cur_y = cur[0]
        cur_prev = prev_dict[cur]
        path.append(cur_prev)
        cur = cur_prev
 
    path.reverse()

    return path;

import heapq
import pdb

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.start
    end = maze.waypoints[0]
    frontier = []
    explored = {}
    explored_step = {}
    path = []
    prev_dict = {}
    rows    = maze.size.y
    columns = maze.size.x
    explored_step[start] = 0
    sentinel = (-1,-1)

    push_pair = (start, sentinel)
    # heapq.heappush(frontier, (0, push_pair))
    # cur = heapq.heappop(frontier)[1][0]
    explored[start] = astar_cost(0, mht_dst(start, end))
    cur = start

    while cur != end:
        cur_neigh = []
        cur_x = cur[1]
        cur_y = cur[0]
        # if(cur_y > 1):
        cur_neigh.append((cur_y-1,cur_x))
        # if(cur_y < rows-2):
        cur_neigh.append((cur_y+1,cur_x))
        # if(cur_x > 1):
        cur_neigh.append((cur_y,cur_x-1))
        # if(cur_x < columns-2):
        cur_neigh.append((cur_y,cur_x+1))
        
        for i in range(len(cur_neigh)):
            checknavi_x = cur_neigh[i][1]
            checknavi_y = cur_neigh[i][0]
            if(maze.navigable(checknavi_y,checknavi_x)):
                neigh_cost = astar_cost(explored_step[cur]+1, mht_dst(cur_neigh[i],end))
                push_pair = (cur_neigh[i], cur)
                if((cur_neigh[i] not in explored) or (explored[cur_neigh[i]] > neigh_cost)):
                    explored[cur_neigh[i]] = neigh_cost
                    prev_dict[cur_neigh[i]] = cur
                    heapq.heappush(frontier, (neigh_cost, push_pair))
                    cur_dict = cur_neigh[i]
                    explored_step[cur_neigh[i]] = explored_step[cur]+1

        cur_pair = heapq.heappop(frontier)
        explored[cur_pair[1][0]] = cur_pair[0]
        cur = cur_pair[1][0]
        prev_dict[cur]=cur_pair[1][1]

    path.append(end);
    cur = end
    cur_x = end[1]
    cur_y = end[0]

    while (cur != start):
        cur_x = cur[1]
        cur_y = cur[0]
        cur_prev = prev_dict[cur]
        path.append(cur_prev)
        cur = cur_prev
 
    path.reverse()

    return path

def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.start
    ends = maze.waypoints
    ends_pair = []
    for i in range(len(ends)):
        for j in range((i+1), len(ends)):
            ends_pair.append((i,j))

    endspair_len_dist = {}
    for pair in ends_pair:
        path_len = len(_astar_single_(maze, ends[pair[0]], ends[pair[1]]))
        endspair_len_dist[pair] = path_len-1

    frontier = []
    prev_dict = {}
    prev_dict[(start, tuple(ends))] = None

    dst_score = {}

    push_pair = (0, (start, tuple(ends)) )
    push_sortkey = astar_cost(0, astar_h(start, tuple(ends), endspair_len_dist, ends) )
    heapq.heappush(frontier, (push_sortkey, push_pair))
    dst_score[(start, tuple(ends))] = 0

    while frontier:
        cur_heap_ele = heapq.heappop(frontier)
        cur_heap_node = cur_heap_ele[1]
        cur = cur_heap_node[1][0]
        if (len(cur_heap_node[1][1]) == 0):
            return form_path(prev_dict, (cur_heap_node[1]))

        cur_x = cur[1]
        cur_y = cur[0]
        cur_neigh = []
        cur_neigh.append((cur_y-1,cur_x))
        cur_neigh.append((cur_y+1,cur_x))
        cur_neigh.append((cur_y,cur_x-1))
        cur_neigh.append((cur_y,cur_x+1))

        for neigh in cur_neigh:
            checknavi_x = neigh[1]
            checknavi_y = neigh[0]
            if(maze.navigable(checknavi_y,checknavi_x)):
                node_left_ends = get_ends_left(neigh, cur_heap_node[1][1])
                node_dst = (neigh, tuple(node_left_ends))
                if ((node_dst in dst_score) and (dst_score[node_dst] <= (dst_score[cur_heap_node[1]] + 1) )):
                    continue
                dst_score[node_dst] = dst_score[cur_heap_node[1]] + 1
                prev_dict[node_dst] = cur_heap_node[1]

                score_1 = cur_heap_ele[0]
                score_2 = dst_score[node_dst] + astar_h(neigh, tuple(node_left_ends), endspair_len_dist, ends)
                push_pair = (dst_score[node_dst], node_dst)
                push_sortkey = max(score_1,score_2)
                heapq.heappush(frontier, (push_sortkey, push_pair))


def fast_h(maze):
    """
    Runs suboptimal search algorithm for extra credit/part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.start
    ends = maze.waypoints
    '''
    ends_pair = []
    for i in range(len(ends)):
        for j in range((i+1), len(ends)):
            ends_pair.append((i,j))

    endspair_len_dist = {}
    for pair in ends_pair:
        path_len = len(_astar_single_h(maze, ends[pair[0]], ends[pair[1]]))
        endspair_len_dist[pair] = path_len-1
        '''

    frontier = []
    prev_dict = {}
    prev_dict[(start, tuple(ends))] = None

    dst_score = {}

    push_pair = (0, (start, tuple(ends)) )
    push_sortkey = astar_h_extra(start, tuple(ends))
    heapq.heappush(frontier, (push_sortkey, push_pair))
    dst_score[(start, tuple(ends))] = 0

    while frontier:
        cur_heap_ele = heapq.heappop(frontier)
        cur_heap_node = cur_heap_ele[1]
        cur = cur_heap_node[1][0]
        if (len(cur_heap_node[1][1]) == 0):
            return form_path(prev_dict, (cur_heap_node[1]))

        cur_x = cur[1]
        cur_y = cur[0]
        cur_neigh = []
        cur_neigh.append((cur_y-1,cur_x))
        cur_neigh.append((cur_y+1,cur_x))
        cur_neigh.append((cur_y,cur_x-1))
        cur_neigh.append((cur_y,cur_x+1))

        for neigh in cur_neigh:
            checknavi_x = neigh[1]
            checknavi_y = neigh[0]
            if(maze.navigable(checknavi_y,checknavi_x)):
                node_left_ends = get_ends_left(neigh, cur_heap_node[1][1])
                node_dst = (neigh, tuple(node_left_ends))
                if ((node_dst in dst_score) and (dst_score[node_dst] <= (dst_score[cur_heap_node[1]] + 1) )):
                    continue
                dst_score[node_dst] = dst_score[cur_heap_node[1]] + 1
                prev_dict[node_dst] = cur_heap_node[1]

                push_sortkey = dst_score[node_dst] + astar_h_extra(neigh, tuple(node_left_ends))
                push_pair = (dst_score[node_dst], node_dst)
                heapq.heappush(frontier, (push_sortkey, push_pair))

def fast(maze):
    """
    Runs suboptimal search algorithm for extra credit/part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.start
    ends = maze.waypoints
    frontier = []
    explored = {}
    explored_step = {}
    path = []
    path.append(start)
    prev_dict = {}
    rows    = maze.size.y
    columns = maze.size.x
    explored_step[start] = 0
    sentinel = (-1,-1) 

    cur = start
    end_tree = MST(ends)
    mst_wt = end_tree.compute_mst_weight()
    explored[start] = nearest_end(ends, cur) + mst_wt

    mst_wt_dict = {}
    for i in maze.indices():
        mst_wt_dict[i] = mst_wt

    while (len(ends) != 0):
        if cur in ends:
            end_idx = ends.index(cur)
            if (len(ends) == 1):
                break
            ends = ends[:end_idx] + ends[end_idx+1:]
            end_tree = MST(ends)
            mst_wt_dict[cur] = end_tree.compute_mst_weight()

            one_path = []
            one_path.append(cur);
            cur_x = cur[1]
            cur_y = cur[0]

            while (cur != start):
                cur_x = cur[1]
                cur_y = cur[0]
                cur_prev = prev_dict[cur]
                if(cur_prev == start):
                    break
                one_path.append(cur_prev)
                cur = cur_prev
            start = one_path[0]
            cur = start
            one_path.reverse()
            for i in one_path:
                path.append(i)
            frontier = []
            explored = {}
            explored_step = {}
            explored_step[start] = 0
            explored[start] = nearest_end(ends, start) + mst_wt_dict[start]

        cur_neigh = []
        cur_x = cur[1]
        cur_y = cur[0]
        cur_neigh.append((cur_y-1,cur_x))
        cur_neigh.append((cur_y+1,cur_x))
        cur_neigh.append((cur_y,cur_x-1))
        cur_neigh.append((cur_y,cur_x+1))

        for neigh in cur_neigh:
            checknavi_x = neigh[1]
            checknavi_y = neigh[0]
            if(maze.navigable(checknavi_y,checknavi_x)):
                neigh_cost = explored_step[cur]+1+ nearest_end(ends, neigh) + mst_wt_dict[cur]
                push_pair = (neigh, cur)

                if((neigh not in explored) or (explored[neigh] > neigh_cost)):
                    explored[neigh] = neigh_cost
                    prev_dict[neigh] = cur
                    heapq.heappush(frontier, (neigh_cost, push_pair))
                    cur_dict = neigh
                    # explored_step[cur_neigh[i]] = explored_step[cur]+1

        cur_pair = heapq.heappop(frontier)
        explored[cur_pair[1][0]] = cur_pair[0]
        cur = cur_pair[1][0]
        prev_dict[cur]=cur_pair[1][1]
        mst_wt_dict[cur] = mst_wt_dict[prev_dict[cur]]
        explored_step[cur] = explored_step[prev_dict[cur]]+1

    one_path = []
    one_path.append(cur)
    while (cur != start):
        cur_x = cur[1]
        cur_y = cur[0]
        cur_prev = prev_dict[cur]
        if(cur_prev == start):
            break
        one_path.append(cur_prev)
        cur = cur_prev

    one_path.reverse()
    for i in one_path:
        path.append(i)

    return path

def mht_dst(node1, node2):
    dist = abs(node1[0]-node2[0]) + abs(node1[1]-node2[1])
    return dist

def astar_cost(g,h):
    return g+h

def nearest_end(ends, start):
    min = 0
    
    if start in ends:
        return 0

    for i in ends:
        if (min == 0):
            min = mht_dst(i, start)
        elif (mht_dst(i, start) < min):
            min = mht_dst(i, start)
    return min

def _astar_single_(maze, new_start, new_end):
    """
    Runs A star for part 3 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = new_start
    end = new_end
    frontier = []
    explored = {}
    explored_step = {}
    path = []
    prev_dict = {}
    rows    = maze.size.y
    columns = maze.size.x
    explored_step[start] = 0
    sentinel = (-1,-1)

    push_pair = (start, sentinel)
    # heapq.heappush(frontier, (0, push_pair))
    # cur = heapq.heappop(frontier)[1][0]
    explored[start] = astar_cost(0, mht_dst(start, end))
    cur = start

    while cur != end:
        cur_neigh = []
        cur_x = cur[1]
        cur_y = cur[0]
        # if(cur_y > 1):
        cur_neigh.append((cur_y-1,cur_x))
        # if(cur_y < rows-2):
        cur_neigh.append((cur_y+1,cur_x))
        # if(cur_x > 1):
        cur_neigh.append((cur_y,cur_x-1))
        # if(cur_x < columns-2):
        cur_neigh.append((cur_y,cur_x+1))
        
        for i in range(len(cur_neigh)):
            checknavi_x = cur_neigh[i][1]
            checknavi_y = cur_neigh[i][0]
            if(maze.navigable(checknavi_y,checknavi_x)):
                neigh_cost = astar_cost(explored_step[cur]+1, mht_dst(cur_neigh[i],end))
                push_pair = (cur_neigh[i], cur)

                if((cur_neigh[i] not in explored) or (explored[cur_neigh[i]] > neigh_cost)):
                    explored[cur_neigh[i]] = neigh_cost
                    prev_dict[cur_neigh[i]] = cur
                    heapq.heappush(frontier, (neigh_cost, push_pair))
                    cur_dict = cur_neigh[i]
                    explored_step[cur_neigh[i]] = explored_step[cur]+1

        cur_pair = heapq.heappop(frontier)
        explored[cur_pair[1][0]] = cur_pair[0]
        cur = cur_pair[1][0]
        prev_dict[cur]=cur_pair[1][1]

    path.append(end);
    cur = end
    cur_x = end[1]
    cur_y = end[0]

    while (cur != start):
        cur_x = cur[1]
        cur_y = cur[0]
        cur_prev = prev_dict[cur]
        path.append(cur_prev)
        cur = cur_prev
 
    path.reverse()

    return path

def _astar_single_h(maze, new_start, new_end):
    start = new_start
    end = new_end
    frontier = []
    explored = []
    explored.append(start)
    prev_dict = {}

    push_sortkey = mht_dst(start, end)
    heapq.heappush(frontier, (push_sortkey, start))

    while frontier:
        cur_heap_ele = heapq.heappop(frontier)
        cur = cur_heap_ele[1]

        if cur == end:
            return form_single_path(start, end, prev_dict)
        
        cur_x = cur[1]
        cur_y = cur[0]
        cur_neigh = []
        cur_neigh.append((cur_y-1,cur_x))
        cur_neigh.append((cur_y+1,cur_x))
        cur_neigh.append((cur_y,cur_x-1))
        cur_neigh.append((cur_y,cur_x+1))

        step = len(form_single_path(start, cur, prev_dict))
        for neigh in cur_neigh:
            checknavi_x = neigh[1]
            checknavi_y = neigh[0]
            if(maze.navigable(checknavi_y,checknavi_x)):
                if neigh not in explored:
                    prev_dict[neigh] = cur
                    push_sortkey = mht_dst(neigh, end) + step
                    heapq.heappush(frontier, (push_sortkey, neigh))
                    explored.append(neigh)

def astar_h (start_node, ends_left, ends_pair_map, ends_all):
    if len(ends_left) == 0:
        return 0
    if len(ends_left) == 1:
        return mht_dst(start_node, ends_left[0])
    tree = MST(ends_left, ends_pair_map, ends_all)
    ends_len = tree.compute_mst_weight()
    find_min_end = []
    for end in ends_left:
        find_min_end.append(mht_dst(start_node, end))
    min = find_min_end[0]
    if (len(find_min_end) != 1):
        for i in range(1, len(find_min_end)):
            if find_min_end[i] < min:
                min = find_min_end[i]

    dist = ends_len + min
    return dist
    
def astar_h_extra (start_node, ends_left):
    if len(ends_left) == 0:
        return 0
    if len(ends_left) == 1:
        return mht_dst(start_node, ends_left[0])
    tree = MST(ends_left)
    ends_len = tree.compute_mst_weight()
    find_min_end = []
    for end in ends_left:
        find_min_end.append(mht_dst(start_node, end))
    # min = min(find_min_end)
    '''if (len(find_min_end) != 1):
        for i in range(1, len(find_min_end)):
            if find_min_end[i] < min:
                min = find_min_end[i]'''

    dist = ends_len + min(find_min_end)
    return dist

def form_path(prev_dict, cur):
    path = []
    while cur != None:
        path.append(cur[0])
        cur = prev_dict[cur]
    path.reverse()
    return path

def form_single_path(start, cur, prev_dict):
    path = []
    while cur != start:
        path.append(cur)
        cur = prev_dict[cur]
    path.append(start)
    path.reverse()
    return path

def get_ends_left(cur, left_ends):
    left = []
    for i in left_ends:
        if (i != cur):
            left.append(i)
    return left
