import sys
import chess
import random
from ex_uct import ChessPlayer
import argparse
import time
from copy import deepcopy


class AbortSimulation(Exception):
    pass


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


class Ccodes:
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'


# helper functions to remove clutter from code
def color(color, s):
    return '{0}{1}{2}'.format(color, s, Bcolors.ENDC)


def red(s):
    return color(Bcolors.FAIL, s)


def blue(s):
    return color(Bcolors.OKBLUE, s)


def green(s):
    return color(Bcolors.OKGREEN, s)


class Chessboard(chess.Bitboard):

    def __init__(self):
        super(Chessboard, self).__init__()
        self.move_count = 0

    def simulate_move(self, move):
        """
        Simulate the given move
        """
        self.move_count += 1
        self.push(move)

    def simulate_moves_from_node(self, node):
        """
        Simulate moves s.t. board reflects the state of node
        """
        for m in node.move_history():
            self.simulate_move(m)

    def reset_simulated_moves(self):
        """
        Undo all simulated moves
        """
        for _ in range(self.move_count):
            self.pop()
        self.move_count = 0


class HumanPlayer(object):
    """
    A simple implementation that asks a human player for the next
    move.
    """
    def __init__(self, board):
        self.board = board

    def inform_move(self, move):
        self.board.push(move)

    def get_next_move(self):
        exit_keywords = ('quit', 'exit')
        valid_move = False
        move = None
        while not valid_move:
            error_str = ''
            sys.stdout.write(green('Your move > '))
            line = sys.stdin.readline().strip()
            if line.lower() in exit_keywords:
                raise AbortSimulation()
            try:
                move = chess.Move.from_uci(line)
                valid_move = self.board.is_legal(move)
                if not valid_move:
                    error_str = '"{}" is not a valid move.'.format(line)
            except ValueError, e:
                error_str = 'Invalid input. Please enter uci strings only'
            if not valid_move:
                print '{0} {1}'.format(red('ERROR:'), error_str)
        yield move


class RandomPlayer(object):
    """
    The easiest possible player. Simply picks one legal move at random.
    """
    def __init__(self, board):
        self.board = board

    def inform_move(self, move):
        self.board.push(move)

    def get_next_move(self):
        yield random.choice([m for m in self.board.legal_moves])


def check_move(move, board):
    """
    Checks if a given move is legal. If not returns a random legal move.
    """
    if board.is_legal(move):
        return move
    else:
        print("Illegal move {}. Random move instead.".format(str(move)))
        return random.choice([m for m in board.legal_moves])


def play_move(board, secs, player):
    """
    Choses the last move generated within the time limit of secs.
    Checks if the move is legal and plays it.
    """
    best_move = None
    start = time.clock()
    move_generator = player.get_next_move()
    try:
        while True:
            tmp_move = next(move_generator)
            if time.clock() - start <= secs:
                best_move = tmp_move
            else:
                break
    except StopIteration:
        pass

    checked_move = check_move(best_move, board)
    board.push(checked_move)
    return board, checked_move


def print_board(board):
    """
    A pretty printer for a board (with field numbers etc.)
    """
    print('')
    board_str = str(board)
    columns = blue(' '.join('abcdefgh'))
    print(columns)
    for idx, line in enumerate(board_str.split('\n')):
        row = 8-idx
        print '{0}  {1}'.format(line, blue(row))
    print('{0}\n'.format(columns))


def simulate_game(players, board, secs, verbose=True):
    current = 'white'
    ply = 0
    result = 'draw'

    try:
        while not board.is_game_over():
            ply += 1
            board, move = play_move(board, secs, players[current])

            if verbose:
                print('{}. {}\'s move: {}'.format(ply, current, move))
                print_board(board)

            if board.is_checkmate():
                result = current
                break

            if board.is_game_over():
                break

            for p in players.values():
                p.inform_move(move)

            current = 'black' if current == 'white' else 'white'

        if verbose:
            print('The game is over. [{}]'.format(result))

    except (KeyboardInterrupt, AbortSimulation):
        print('\nExiting chess simulation')

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", help="Play against a random player",
                        action="store_true")
    parser.add_argument("--human", help="Play against a human",
                        action="store_true")
    parser.add_argument("--white", help="Play white", action="store_true")
    parser.add_argument("--black", help="Play black", action="store_true")
    parser.add_argument("--secs", help="Seconds each player has for each turn",
                        type=int, default=2)
    args = parser.parse_args()

    print("""
Welcome to the Chess-Simulator. This simulator assumes you have already
implemented the algorithm in the corresponding ex_uct.py file.
Use "exit" or Ctrl-C to quit the chess simulation.
        """)
    if args.human:
        print("""All commands are read from stdin.
    Please enter your moves in UCI syntax, such as "h2h3".
        """)

    board = Chessboard()
    opponent = HumanPlayer(deepcopy(board)) if args.human else RandomPlayer(deepcopy(board))

    if args.black:
        white = opponent
        black = ChessPlayer(deepcopy(board), chess.BLACK)
    else:
        white = ChessPlayer(deepcopy(board), chess.WHITE)
        black = opponent

    players = {'white': white, 'black': black}

    print("Players:")
    print("black: {}".format(type(players['black']).__name__))
    print("WHITE: {}\n".format(type(players['white']).__name__))

    winner = simulate_game(players, board, args.secs)


if __name__ == '__main__':
    main()
