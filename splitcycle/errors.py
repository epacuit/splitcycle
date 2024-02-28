'''Macros for common error messages'''

def not_enough_candidates():
    '''Raised when candidates do not align with ballots'''
    raise ValueError(
        'Number of candidates in `ballots` does not match number of '
        'candidates in `candidates` (i.e. some ranked candidates '
        'could not be matched with names from provided data)'
    )
