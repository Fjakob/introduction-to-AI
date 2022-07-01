import chess


def evaluation(board):
    """
    This is an evaluation of the famous evaluation function of Claude
    Shannon from his paper "Programming a Computer for Playing Chess",
    Philosophical Magazine, Ser.7, Vol. 41, No. 314 - March 1950.

    This is by no means a very good evaluation function, and Shannon notes
    that on his paper ("The coefficients .5 and .1 are merely the writer's
    rough estimate. Furthermore, there are many other terms that should be
    included [5]. The formula is given only for illustrative purposes.")

    It still gives some notion of value for a position and together with UCT
    it still performs not too bad.

    The value is positive if white is in favor and negative if black is in
    favor at a given position.

    :param board: The board to evaluate
    :param player: The player to evaluate for
    :return: A score for the board
    """
    score = 0
    # Material on the board
    for p in range(64):
        piece = board.piece_at(p)

        if piece == chess.Piece.from_symbol('p'):
            score -= 1
        if piece == chess.Piece.from_symbol('r'):
            score -= 5
        if piece == chess.Piece.from_symbol('n'):
            score -= 3
        if piece == chess.Piece.from_symbol('b'):
            score -= 3
        if piece == chess.Piece.from_symbol('q'):
            score -= 9
        if piece == chess.Piece.from_symbol('k'):
            score -= 200

        if piece == chess.Piece.from_symbol('P'):
            score += 1
        if piece == chess.Piece.from_symbol('R'):
            score += 5
        if piece == chess.Piece.from_symbol('N'):
            score += 3
        if piece == chess.Piece.from_symbol('B'):
            score += 3
        if piece == chess.Piece.from_symbol('Q'):
            score += 9
        if piece == chess.Piece.from_symbol('K'):
            score += 200

    if board.turn == chess.BLACK:
        invert = -1
    else:
        invert = 1

    # Mobility
    score += (invert * .1 * board.pseudo_legal_move_count())
    board.push(None)
    score -= (invert * .1 * board.pseudo_legal_move_count())

    # doubled and isolated pawns
    score -= (.5 * di_pawns(board, chess.WHITE))
    score += (.5 * di_pawns(board, chess.BLACK))

    board.pop()
    return score


def di_pawns(board, player):
    """
    Calculates the number of doubled and isolated pawns for a given
    player.
    :param board:
    :param player:
    :return:
    """
    doubled = 0
    isolated = 0
    number_of_pawns = [0] * 10

    if player == chess.WHITE:
        pawn = chess.Piece.from_symbol('P')
    else:
        pawn = chess.Piece.from_symbol('p')

    for i, f in enumerate(chess.FILE_NAMES):
        number_of_pawns[i+1] = 0
        for r in range(1, 9):
            if board.piece_at(chess.SQUARE_NAMES.index(f + str(r))) == pawn:
                number_of_pawns[i+1] += 1
        if number_of_pawns[i+1] > 1:
            doubled += 1

    for i in range(1,9):
        if number_of_pawns[i] == 1 and number_of_pawns[i-1] == 0 and number_of_pawns[i+1] == 0:
            isolated += 1

    return doubled + isolated
