import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if depth == 0:
      return (evaluate(board),[],{})

    else:
      moves_list = []
      val_list = []
      moves_tree = {}
      moves = [ move for move in generateMoves(side, board, flags) ]
      min_val = float('inf')
      max_val = -float('inf')
      for move in moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        temp_val = minimax(newside, newboard, newflags, depth-1)
        moves_tree[encode(*move)] = temp_val[2]
        val_list.append(temp_val)
        if side == True:
          if min_val > temp_val[0]:
            min_val = temp_val[0]
            min_node_list = list(temp_val[1])
            min_node_list.insert(0, move)
        elif side == False:
          if max_val < temp_val[0]:
            max_val = temp_val[0]
            max_node_list = list(temp_val[1])
            max_node_list.insert(0, move)

      if side == True:
        return (min_val, min_node_list, moves_tree)
      elif side == False:
        return (max_val, max_node_list, moves_tree)

def min_value(side, board, flags):
    min_val = float('inf')
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) <= 0:
        return (evaluate(board), [], {})
    else:
        for temp_move in moves:
          newside, newboard, newflags = makeMove(side, board, temp_move[0], temp_move[1], flags, temp_move[2])
          temp_value = evaluate(newboard)
          if temp_value < min_val:
            min_val = temp_value
            return_tuple = (temp_value, [ temp_move ], { encode(*temp_move): {} })
    return return_tuple

def max_value(side, board, flags):
    max_val = -float('inf')
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) <= 0:
        return (evaluate(board), [], {})
    else:
        for temp_move in moves:
          newside, newboard, newflags = makeMove(side, board, temp_move[0], temp_move[1], flags, temp_move[2])
          temp_value = evaluate(newboard)
          if temp_value > max_val:
            max_val = temp_value
            return_tuple = (temp_value, [ temp_move ], { encode(*temp_move): {} })
    return return_tuple
        

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if depth <= 0:
      return (evaluate(board),[],{})
    
    else:
      min_node_list = []
      max_node_list = []
      moves_tree = {}
      moves = [ move for move in generateMoves(side, board, flags) ]
      if side == True:
        if len(moves) <= 0:
          return (evaluate(board), [], {})
        for move in moves:
          newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
          temp_val = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)
          moves_tree[encode(*move)] = temp_val[2]
          if beta > temp_val[0]:
            beta = temp_val[0]
            min_node_list = list(temp_val[1])
            min_node_list.insert(0, move)
          if beta <= alpha:
            break
        return (beta, min_node_list, moves_tree)
      elif side == False:
        if len(moves) <= 0:
          return (evaluate(board), [], {})
        for move in moves:
          newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
          temp_val = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)
          moves_tree[encode(*move)] = temp_val[2]
          if alpha < temp_val[0]:
            alpha = temp_val[0]
            max_node_list = list(temp_val[1])
            max_node_list.insert(0, move)
          if beta <= alpha:
            break
        return (alpha, max_node_list, moves_tree) 

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    if depth == 0:
      return (evaluate(board),[],{})

    else:
      moves_list = []
      moves_tree = {}
      moves = [ move for move in generateMoves(side, board, flags) ]
      min_val = float('inf')
      max_val = -float('inf')
      if len(moves) <= 0:
        return (evaluate(board),[],{})
      for move in moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        temp_val = _stochastic(newside, newboard, newflags, depth-1, breadth, chooser, 1)
        moves_tree[encode(*move)] = temp_val[2]
        if side == True:
          if min_val > temp_val[0]:
            min_val = temp_val[0]
            min_node_list = list(temp_val[1])
            min_node_list.insert(0, move)
        elif side == False:
          if max_val < temp_val[0]:
            max_val = temp_val[0]
            max_node_list = list(temp_val[1])
            max_node_list.insert(0, move) 

      if side == True:
        print("min trees", moves_tree)
        return(min_val, min_node_list, moves_tree)
        # else:
          # return (total_val/breadth, min_node_list, moves_tree)
      elif side == False:
        return(max_val, max_node_list, moves_tree)
        # else:
          # return (total_val/breadth, max_node_list, moves_tree)

def _stochastic(side, board, flags, depth, breadth, chooser, choice_flag):
    if depth == 0:
      return (evaluate(board),[],{})

    else:
      moves_list = []
      moves_tree = {}
      moves = [ move for move in generateMoves(side, board, flags) ]
      min_val = float('inf')
      max_val = -float('inf')
      total_val = 0
      if len(moves) <= 0:
        return (evaluate(board),[],{})
      if choice_flag == 1:
        choice_flag = 0
        for i in range(breadth):
          move = chooser(moves)
          newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
          temp_val = _stochastic(newside, newboard, newflags, depth-1, breadth, chooser, choice_flag)
          moves_tree[encode(*move)] = temp_val[2]
          total_val += temp_val[0]
          if side == True:
            min_node_list = list(temp_val[1])
            min_node_list.insert(0, move)
          elif side == False:
            max_node_list = list(temp_val[1])
            max_node_list.insert(0, move) 
        if side == True:
          return (total_val/breadth, min_node_list, moves_tree)
        elif side == False:
          return (total_val/breadth, max_node_list, moves_tree)
      elif choice_flag == 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        temp_val = _stochastic(newside, newboard, newflags, depth-1, breadth, chooser, choice_flag)
        moves_tree[encode(*move)] = temp_val[2]
        if side == True:
          min_node_list = list(temp_val[1])
          min_node_list.insert(0, move)
          return (temp_val[0], min_node_list, moves_tree)
        elif side == False:
          max_node_list = list(temp_val[1])
          max_node_list.insert(0, move)
          return (temp_val[0], max_node_list, moves_tree)

