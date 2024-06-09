import numpy as np
# import matplotlib.pyplot as plt


""" Player Functions """

def playerCardCounterUpdate( running_score : float, num_cards_left_in_shoe : int, observed_card : int) -> float:
    # updates the card Counting signal based on:
    #  card_counting_signal_input : the previous card counting signal (which is a float)
    #  observed_card : the card that was dealt out that we just observed
    #  num_cards_left_in_shoe : how many cards are currently left in the deck

    # If the current card_counting_signal is None, then we are just starting a new shoe and must initialize it to some value
    if running_score == None:
        return 0

    # Updating the running score
    if observed_card < 0: # [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]
       running_score -= 2
    elif observed_card <= 7: # [1, 2, 3, 4, 5, 6, 7]
       running_score += 13/7
    elif observed_card >= 8: # [8, 9, 10]
       running_score -= 1
  
    return running_score

# Maps a card_counting signal to an index
def calculateCardCountingIndex ( card_counting_signal_input : float ) -> int:
    inp = round(card_counting_signal_input)

    if inp <= -30:
        return 0
    elif inp <= -20:
        return 1
    elif inp <= -10:
       return 2
    elif inp <= -5:
        return 3
    elif inp <= 5:
        return 4
    elif inp <= 10:
        return 5
    elif inp <= 20:
        return 6
    elif inp <= 30:
        return 7
    elif inp <= 40:
       return 8
    else:
       return 9

def playerBetSizeChoice(card_counting_signal_input:float, num_cards_left_in_shoe : int, current_bankroll:int) -> int:
    # How much the player wants to bet
    # Is allowed to depend on the card counting signal, the number of cards left in the deck, and the plyaers current bankroll
    # The actual bet must obey the following rules (which are enfored by the simulation):
    # - bet_size is an integer which is >= 0 (betting 0 is allowed)
    # - bet_size is <= 100,000 (this is the table maximum bet allowed!)
    # - bet_size is <= current_bankroll (you cannot bet more than you have)

    probabilty_values =    [0.        ,0.        ,
                            0.45255474,0.48399927,
                            0.50250821,0.52633729,
                            0.55386211,0.56927797,
                            0.59459459,0.        ]

    
    card_counting_signal_index = calculateCardCountingIndex(card_counting_signal_input)
    proportion = (probabilty_values[card_counting_signal_index] 
                  + 1.2 * (probabilty_values[card_counting_signal_index] - 0.5) 
                  + 0.011 * (10 - (num_cards_left_in_shoe // 30)))

    if proportion > 0.95:
        proportion = 0.95
        
    return int(current_bankroll * proportion)


def playerStrategy(player_sum, dealer_sum,card_counting_signal) -> bool:
  # Whether to hit or stick given the player sum, dealer sum and card counting signal (True/1 = Hit, False/0 = Stick)

  actions = np.array(
     [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
  
  return actions[dealer_sum][player_sum]




""" Learning functions"""
# Q-Learning (off-policy TD control) for estimating the policy (Hit/Stick)
def learnStrategy(playerCardCounterUpdate, numCardCountCountValues=10):
    # Array Initialization
    #                  Card Counting Input,Action, Dealer, Player
    q_func = np.zeros((numCardCountCountValues, 2, 22, 22), dtype=float)
    policy_array = np.zeros((numCardCountCountValues, 22, 22), dtype=int)

    # Epsilon Greedy Policy
    def gamblers_problem_epsilon_greedy(dealer_sum, player_sum, player_cardcount_signal, epsilon):
        probability_epsilon_event = bool(np.random.rand() < epsilon)
        player_cardcount_signal_index = calculateCardCountingIndex(player_cardcount_signal)

        if probability_epsilon_event: # Selecting a purely random action (Hit/Stick)
            return int(np.random.randint(2))
        else: # Doing a purely greedy action choice
            return policy_array[player_cardcount_signal_index][dealer_sum][player_sum]
    
    # Setting value for terminal states
    q_func[:][:] = [[-1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1., 0., 1., 1., 1., 1.], #17
                    [-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1., 0., 1., 1., 1.],
                    [-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1., 0., 1., 1.],
                    [-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1., 0., 1.],
                    [-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1., 0.]]

    # Game simulation Constants
    NUM_ONE_DECKS_IN_SHOE = 10 # number of ONE_DECKs in the shoe at the start of the game
    MIN_ONE_DECKS_LEFT_TO_END_GAME = 1 # when the shoe is low, the next hand is not dealt

    # Setting up ONE_DECK
    CARDS_1_TO_10 = np.arange(1, 10 + 1)
    ONE_DECK = np.concatenate( (CARDS_1_TO_10, CARDS_1_TO_10, -CARDS_1_TO_10) ) #the ONE_DECK is 30 cards, 1-10 twice and -1 to 10 once

    # Running Episodes
    max_shoes =  10000
    epsilon = 0.5
    learning_rate = 0.02

    for shoes in range(max_shoes):
        # Displaying training progress
        if shoes % 1000 == 0:
            print(f"{shoes=}")

        # Updating learning and exploration rates
        if shoes > (max_shoes/3): 
            learning_rate = 0.01
            epsilon = 0.05

        if shoes > 2*(max_shoes/3): 
            learning_rate = 0.005
            epsilon = 0

        shoe = np.tile(ONE_DECK, NUM_ONE_DECKS_IN_SHOE) #the shoe is the full stack of cards used in the game
        np.random.shuffle(shoe) #shuffle the ONE_DECK!

        top_of_shoe_ix = 0 #index for how far we are in the shoe
        num_cards_left_in_shoe = len(shoe) #number of cards remaining in the shoe
        player_cardcount_signal = float(playerCardCounterUpdate(None, None, None)) #get the starting cardcount signal by feeding "None" into the updater
        
        ## PLAY ANOTHER HAND LOOP
        # while we still have at least MIN_ONE_DECKS_LEFT_TO_END_GAME left in the shoe, the game continues
        while len(shoe) - top_of_shoe_ix > MIN_ONE_DECKS_LEFT_TO_END_GAME*len(ONE_DECK):
            round_seen_player_states = []

            # Deal the next card to the player
            nextCard = shoe[top_of_shoe_ix]
            top_of_shoe_ix += 1
            num_cards_left_in_shoe -= 1

            cardcount_index = calculateCardCountingIndex(player_cardcount_signal)

            # Feed in the card to the cardcounter function
            player_cardcount_signal = float(playerCardCounterUpdate(player_cardcount_signal, num_cards_left_in_shoe, nextCard))

            #for the 1st card only, we take the absolute value (since we always start at a positive value)
            player_sum = abs(nextCard)
            temp_player_sum = player_sum

            # Deal the next card to the dealer and also let the player card counter see the next card
            nextCard = shoe[top_of_shoe_ix]
            top_of_shoe_ix += 1
            num_cards_left_in_shoe -= 1
            player_cardcount_signal = float(playerCardCounterUpdate(player_cardcount_signal,num_cards_left_in_shoe,nextCard))
            player_cardcount_signal_index = calculateCardCountingIndex(player_cardcount_signal)
            dealer_sum = abs(nextCard) # Absolute value is taken for the dealer starting card

            player_is_active = gamblers_problem_epsilon_greedy(player_sum, dealer_sum, player_cardcount_signal, epsilon)
            player_is_active = player_is_active and ( top_of_shoe_ix < MIN_ONE_DECKS_LEFT_TO_END_GAME*len(ONE_DECK) )
            player_is_busted = False # Boolean flag for the player being busted.


            # Playing a players turn until their policy decides to stop or the game is over
            while player_is_active:
                #deal the next card and also let the player card counter see the next card


                nextCard = shoe[top_of_shoe_ix]
                top_of_shoe_ix += 1
                num_cards_left_in_shoe -= 1
                player_cardcount_signal = float(playerCardCounterUpdate(player_cardcount_signal,num_cards_left_in_shoe,nextCard))
                player_cardcount_signal_index = calculateCardCountingIndex(player_cardcount_signal)
                player_sum += nextCard

                #check status for bustedness and what to do next
                player_is_busted = ( player_sum < 1 or player_sum > 21 ) #flag for busted
                player_is_active = ( not player_is_busted ) and gamblers_problem_epsilon_greedy(player_sum, dealer_sum, player_cardcount_signal, epsilon)#check if player is still active
                player_is_active = player_is_active and ( top_of_shoe_ix < MIN_ONE_DECKS_LEFT_TO_END_GAME*len(ONE_DECK) ) #if we are at out of cards in the shoe everything is automatically over!

                if not player_is_busted:
                    round_seen_player_states.append((temp_player_sum, cardcount_index, player_sum, player_cardcount_signal_index, player_is_active, dealer_sum))
                    
                    temp_player_sum = player_sum
                    cardcount_index = calculateCardCountingIndex(player_cardcount_signal)

            #The dealer will always hit if <=17 and player is not busted
            dealer_is_active = (not player_is_busted) #flag for the Dealer still playing. Note that if player busted, the dealer will just skip their turn.
            dealer_is_active = dealer_is_active and ( top_of_shoe_ix < MIN_ONE_DECKS_LEFT_TO_END_GAME*len(ONE_DECK) ) #if we are out of cards everything is automatically over
            dealer_is_busted = False

            temp_dealer_sum = dealer_sum

            while dealer_is_active:
                #deal card to dealer
                nextCard = shoe[top_of_shoe_ix]
                top_of_shoe_ix += 1
                num_cards_left_in_shoe -= 1
                player_cardcount_signal = float(playerCardCounterUpdate(player_cardcount_signal,num_cards_left_in_shoe,nextCard))
                dealer_sum += nextCard

                #check for what to do
                dealer_is_busted = (dealer_sum < 1) or (dealer_sum > 21) #check if dealer busted
                dealer_is_active = (not dealer_is_busted) and ( dealer_sum <= 16 )#check if dealer chooses to hit and keep playing
                dealer_is_active = dealer_is_active and ( top_of_shoe_ix < MIN_ONE_DECKS_LEFT_TO_END_GAME*len(ONE_DECK) ) #if we are out of cards everything is automatically over

            if dealer_is_busted:
                dealer_sum = 0

            # Updating values as per the update rule
            q_func[cardcount_index][int(player_is_busted)][temp_dealer_sum][temp_player_sum] += (learning_rate * 
                                    ( np.maximum(q_func[cardcount_index][1][dealer_sum][temp_player_sum], q_func[cardcount_index][1][dealer_sum][temp_player_sum])
                                    - q_func[cardcount_index][int(player_is_busted)][temp_dealer_sum][temp_player_sum]))
            
            policy_array[cardcount_index][temp_dealer_sum][temp_player_sum] = np.argmax(np.array([q_func[cardcount_index][action][dealer_sum][temp_player_sum] for action in range(2)]))


            for temp_player_sum, cardcount_index, player_sum_after, player_cardcount_signal_index_after, action_after, dealer_sum in round_seen_player_states[::-1]:
                q_func[cardcount_index][1][dealer_sum][temp_player_sum] += (learning_rate * 
                                    ( np.maximum(q_func[player_cardcount_signal_index_after][0][dealer_sum][player_sum_after], q_func[player_cardcount_signal_index_after][1][dealer_sum][player_sum_after])
                                    - q_func[cardcount_index][1][dealer_sum][temp_player_sum]))
                policy_array[cardcount_index][temp_dealer_sum][temp_player_sum] = np.argmax(np.array([q_func[cardcount_index][action][dealer_sum][temp_player_sum] for action in range(2)]))

    return policy_array

# Q-Learning (off-policy TD control) for learning optimal bet sizes
def learnBetSize(playerStrategy, playerCardCounterUpdate, probabilty_values):

    # The k value from write up equation bet_choice
	prob_multipler_options = np.arange(0, 2.1, 0.1)
	
    # The w value from write up equation bet_choice
	deck_multipler_options = np.arange(0.000, 0.03, 0.001)
    	
    # To use in the local
	function_vals = np.array([0.0, 0.0])

    # Value function of size prob_multipler_options
	value_func = np.zeros((np.size(prob_multipler_options), np.size(deck_multipler_options)))


    # bet_choice function updated on the function_vals
	def bet_choice(card_counting_signal_input:float, num_cards_left_in_shoe : int, current_bankroll:int) -> int:
        
		card_counting_signal_index = calculateCardCountingIndex(card_counting_signal_input)
		proportion = probabilty_values[card_counting_signal_index] + function_vals[0] * (probabilty_values[card_counting_signal_index] - 0.5) + function_vals[1] * (10 - (num_cards_left_in_shoe // 30))

		if proportion > 0.95:
			proportion = 0.95
        
		return int(current_bankroll * proportion)
    
	def gamblers_problem_epsilon_greedy(epsilon):

		probability_epsilon_event = bool(np.random.rand() < epsilon)

		if probability_epsilon_event: # Selecting a purely random k and w
            
			# Choose a random Index
			prob_multi_index = np.random.randint(np.size(prob_multipler_options))
			deck_multi_index = np.random.randint(np.size(deck_multipler_options))
            
			# Set function to the values at those indexs
			function_vals[0] = prob_multipler_options[prob_multi_index]
			function_vals[1] = deck_multipler_options[deck_multi_index]
            
			# return the indexes to the funciton for learning
			return np.array([prob_multi_index, deck_multi_index])

		else: # Doing a purely greedy bet size (selecting the action that maximizes value)
            
			# Choose the highest values in the (k,w) pair
			index = np.unravel_index(np.argmax(value_func, axis=None), value_func.shape)
            
			# Set function to the values at those indexs
			function_vals[0] = prob_multipler_options[index[0]]
			function_vals[1] = deck_multipler_options[index[1]]
            
			# return the indexes to the funciton for learning
			return index


	epsilon = 0.5
	learning_rate = 0.5
	num_plays = 10000

	for episode_num in range(num_plays):

		if episode_num % 100 == 0:
				print(f"{episode_num=}")

        # Adjusting the learning rate and epsilon
		if episode_num > 2500: 
			learning_rate = 0.1
			epsilon = 0.1

		if episode_num > 7500: 
			learning_rate = 0.05
			epsilon = 0.

		# Get k and w from epsilon greedy value
		index = gamblers_problem_epsilon_greedy(epsilon)

		# Run 10 full games
		game_total = 0
		for _ in range(10):
			game_total += simulateEasy21_finite_deck(playerStrategy,bet_choice,playerCardCounterUpdate)

		# reward is what is left over after initlial investments of 100*10
		reward = game_total - 1000

		# Learn the value of given state
		value_func[index[0]][index[1]] += learning_rate * (reward - value_func[index[0]][index[1]])
    	
	# At the end, return the value with most value
	index = np.unravel_index(np.argmax(value_func, axis=None), value_func.shape)
	function_vals[0] = prob_multipler_options[index[0]]
	function_vals[1] = deck_multipler_options[index[1]]

	return value_func, function_vals


def learn_probability_to_win_hand(player_policy, card_counting_total_indexes):

  #player_policy is the policy being used by the player (e.g. Lab2 policy)
  
  # Initializing arrays for calculations
  number_times_shoes_seen = np.zeros(card_counting_total_indexes)
  number_times_shoes_won = np.zeros(card_counting_total_indexes)
  number_times_shoes_seen_cardleft = np.zeros((10,card_counting_total_indexes))
  number_times_shoes_won_cardleft = np.zeros((10,card_counting_total_indexes))
  probability_player_wins_hand = np.zeros(card_counting_total_indexes)
  probability_player_wins_hand_cardleft = np.zeros((10,card_counting_total_indexes))

  NUM_ONE_DECKS_IN_SHOE = 10 # number of ONE_DECKs in the shoe at the start of the game
  MIN_ONE_DECKS_LEFT_TO_END_GAME = 1 # when the shoe is low, the next hand is not dealt

  # Setting up ONE_DECK
  CARDS_1_TO_10 = np.arange(1, 10 + 1)
  ONE_DECK = np.concatenate( (CARDS_1_TO_10, CARDS_1_TO_10, -CARDS_1_TO_10) ) #the ONE_DECK is 30 cards, 1-10 twice and -1 to 10 once

  max_shoes = 10000
  for shoes in range(max_shoes):
    
    # Printing learning progress for user
    if shoes % 100 == 0:
      print(f"{shoes=}")

    shoe = np.tile(ONE_DECK, NUM_ONE_DECKS_IN_SHOE) #the shoe is the full stack of cards used in the game
    np.random.shuffle(shoe) #shuffle the ONE_DECK!

    top_of_shoe_ix = 0 #index for how far we are in the shoe
    num_cards_left_in_shoe = len(shoe) #number of cards remaining in the shoe
    player_cardcount_signal = float(playerCardCounterUpdate(None, None, None)) #get the starting cardcount signal by feeding "None" into the updater

    ## PLAY ANOTHER HAND LOOP
    # while we still have at least MIN_ONE_DECKS_LEFT_TO_END_GAME left in the shoe, the game continues
    while len(shoe) - top_of_shoe_ix > MIN_ONE_DECKS_LEFT_TO_END_GAME*len(ONE_DECK):

      # Get the current_player_cardcount_signal before round starts
      current_player_cardcount_signal = player_cardcount_signal
      current_shoe_index = int(round(current_player_cardcount_signal, 1) * 10) + 100
      number_times_shoes_seen[current_shoe_index] += 1

      ### DEAL INITIAL CARDS TO DEALER AND PLAYER ###

      # Deal the next card to the player
      nextCard = shoe[top_of_shoe_ix]
      top_of_shoe_ix += 1
      num_cards_left_in_shoe -= 1
      number_times_shoes_seen_cardleft[(num_cards_left_in_shoe // 30)][current_shoe_index] += 1

      # Feed in the card to the cardcounter function
      player_cardcount_signal = float(playerCardCounterUpdate(player_cardcount_signal, num_cards_left_in_shoe, nextCard))

      #for the 1st card only, we take the absolute value (since we always start at a positive value)
      player_sum = abs(nextCard)

      # Deal the next card to the dealer and also let the player card counter see the next card
      nextCard = shoe[top_of_shoe_ix]
      top_of_shoe_ix += 1
      num_cards_left_in_shoe -= 1
      player_cardcount_signal = float(playerCardCounterUpdate(player_cardcount_signal,num_cards_left_in_shoe,nextCard))
      dealer_sum = abs(nextCard) # Absolute value is taken for the dealer starting card

      player_is_active = player_policy(player_sum, dealer_sum, player_cardcount_signal)
      player_is_active = player_is_active and ( top_of_shoe_ix < len(shoe) )
      player_is_busted = False # Boolean flag for the player being busted.

      # Playing a players turn until their policy decides to stop or the game is over
      while player_is_active:
        #deal the next card and also let the player card counter see the next card
        nextCard = shoe[top_of_shoe_ix]
        top_of_shoe_ix += 1
        num_cards_left_in_shoe -= 1
        player_cardcount_signal = float(playerCardCounterUpdate(player_cardcount_signal,num_cards_left_in_shoe,nextCard))
        player_sum += nextCard

        #check status for bustedness and what to do next
        player_is_busted = ( player_sum < 1 or player_sum > 21 ) #flag for busted
        player_is_active = ( not player_is_busted ) and player_policy(player_sum, dealer_sum, player_cardcount_signal)#check if player is still active
        player_is_active = player_is_active and ( top_of_shoe_ix < len(shoe) ) #if we are at out of cards in the shoe everything is automatically over!

      #The dealer will always hit if <=17 and player is not busted
      dealer_is_active = (not player_is_busted) #flag for the Dealer still playing. Note that if player busted, the dealer will just skip their turn.
      dealer_is_active = dealer_is_active and ( top_of_shoe_ix < len(shoe) ) #if we are out of cards everything is automatically over
      dealer_is_busted = False

      while dealer_is_active:
        #deal card to dealer
        nextCard = shoe[top_of_shoe_ix]
        top_of_shoe_ix += 1
        num_cards_left_in_shoe -= 1
        player_cardcount_signal = float(playerCardCounterUpdate(player_cardcount_signal,num_cards_left_in_shoe,nextCard))
        dealer_sum += nextCard

        #check for what to do
        dealer_is_busted = (dealer_sum < 1) or (dealer_sum > 21) #check if dealer busted
        dealer_is_active = (not dealer_is_busted) and ( dealer_sum <= 16 )#check if dealer chooses to hit and keep playing
        dealer_is_active = dealer_is_active and ( top_of_shoe_ix < len(shoe) ) #if we are out of cards everything is automatically over

      #shoe value index
      player_wins = dealer_is_busted or ( (not player_is_busted) and player_sum > dealer_sum ) #Boolean variable if player wins!
      if player_wins:
        
		# learn probabilty for current_shoe_index
        number_times_shoes_won[current_shoe_index] += 1
        
		# learn probabilty for current_shoe_index and num_cards_left_in_shoe
        number_times_shoes_won_cardleft[(num_cards_left_in_shoe // 30)][current_shoe_index] += 1


  # Calculate average win rate for each value of shoe_average seen
  for i in range(card_counting_total_indexes):
    if number_times_shoes_seen[i] > 1:
      probability_player_wins_hand[i] = number_times_shoes_won[i]/number_times_shoes_seen[i]
    for n in range(10):
      if (number_times_shoes_seen_cardleft[n][i]) > 1:
        probability_player_wins_hand_cardleft[n][i] = number_times_shoes_won_cardleft[n][i]/number_times_shoes_seen_cardleft[n][i]

  return probability_player_wins_hand, number_times_shoes_seen, probability_player_wins_hand_cardleft


# # Use this code to graph
# # Make sure to import matplotlib!

# # Current submitted card counting function has 10 indexes, the averge value card counting will have 201 indexes.
# card_counting_total_indexes = 10
# player_win_prob, number_times_shoes_seen, probability_player_wins_hand_cardleft = learn_probability_to_win_hand(playerStrategy, card_counting_total_indexes)

# fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# axs[0].plot(np.arange(0,201), number_times_shoes_seen, linestyle='', marker='.')
# axs[0].set_title('number_times_shoes_seen')

# axs[1].plot(np.arange(0,201), player_win_prob, linestyle='', marker='.')
# axs[1].set_title('player_win_prob')
# axs[1].set_ylim(0.0, 1.0)

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Show the plots
# plt.show()

# fig, axs = plt.subplots(2, 5, figsize=(20, 6))

# for i in range(0, 5):
#     axs[0,i].plot(np.arange(0,201), probability_player_wins_hand_cardleft[i], linestyle='', marker='.')
#     axs[0,i].set_title(f'player_win_prob when {i} decks remain')
#     axs[0,i].set_ylim(0.0, 1.0)

# for i in range(5, 10):
#     axs[1,i-5].plot(np.arange(0,201), probability_player_wins_hand_cardleft[i], linestyle='', marker='.')
#     axs[1,i-5].set_title(f'player_win_prob when {i} decks remain')
#     axs[1,i-5].set_ylim(0.0, 1.0)

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Show the plots
# plt.show()


""" Simulation (this is what the leaderboard runs!) """
def simulateEasy21_finite_deck(playerStrategy, playerBetSizeChoice, playerCardCounterUpdate, Verbose = False) -> int:
  # simulates one run through of a shoe of easy 21 using the player and delaer strategy
  # the game is counted as over when there are <MIN_ONE_DECKS_LEFT_TO_END_GAME worth of cards left in the shoe

  ####GAME PARAMETERS#####
  # These parameters set up the "rules" of the game
  # They are constant
  NUM_ONE_DECKS_IN_SHOE = 10 #number of Easy21 decks in the shoe at the start of the game
  MIN_ONE_DECKS_LEFT_TO_END_GAME = 1 #when the shoe is below this amount, the next hand is not dealt
  PLAYER_STARTING_BANKROLL = 100 #staring bankroll for hte player
  TABLE_MAXIMUM_BET_SIZE = 100_000 #maximum bet size for any single bet
  #####################

  CARDS_1_TO_10 = np.arange(1,10+1)
  ONE_DECK = np.concatenate( (CARDS_1_TO_10, CARDS_1_TO_10, -CARDS_1_TO_10) ) #the ONE_DECK is 30 cards, 1-10 twice and -1 to 10 once

  shoe = np.tile( ONE_DECK, NUM_ONE_DECKS_IN_SHOE) #the shoe is the full stack of cards used in the game
  np.random.shuffle(shoe) #shuffle the ONE_DECK!

  top_of_shoe_ix = 0 #index for how far we are in the shoe
  num_cards_left_in_shoe = len(shoe) - top_of_shoe_ix #number of cards remaining in the shoe

  player_bankroll = PLAYER_STARTING_BANKROLL   #your starting bankroll initizled to the start value
  player_savings_account = 0 #starting savings account amount
  player_cardcount_signal = float(playerCardCounterUpdate(None,None,None)) #get the starting cardcount signal by feeding "None" into the updater

  #### PLAY ANOTHER HAND LOOP
  # while we still have at least MIN_ONE_DECKS_LEFT_TO_END_GAME left in the shoe, the game continues
  # (and the player has a non-zero bankroll!)
  while len(shoe) - top_of_shoe_ix > MIN_ONE_DECKS_LEFT_TO_END_GAME*len(ONE_DECK) and player_bankroll > 0:

    current_round_cardcount_signal = player_cardcount_signal

    ### PLAYER CHOOSES BETSIZE ####
    bet_size = playerBetSizeChoice(player_cardcount_signal, num_cards_left_in_shoe, player_bankroll) #choose the betsize!
    bet_size = int(  np.clip( bet_size , a_min = 0, a_max = min(player_bankroll,TABLE_MAXIMUM_BET_SIZE) ) ) #ensure the playerbet size follows the rules, integer >=0, and cannot exceed player bankroll or maximum bet size.

    #print out whats going on if requested
    if Verbose:
      print(f"{player_bankroll=}, {bet_size=}")

    ### DEAL INITIAL CARDS TO DEALER AND PLAYER ###

    #deal the next card to the player
    nextCard = shoe[top_of_shoe_ix]
    top_of_shoe_ix += 1
    num_cards_left_in_shoe = len(shoe) - top_of_shoe_ix

    #feed in the card to the cardcounter function
    player_cardcount_signal = float(playerCardCounterUpdate(player_cardcount_signal,num_cards_left_in_shoe,nextCard))

    #for the 1st card only, we take the absolute value (since we always start at a positive value)
    player_sum = abs(nextCard) #absolute value is taken for the starting card

    #deal the next card to the dealer and also let the player card counter see the next card
    nextCard = shoe[top_of_shoe_ix]
    top_of_shoe_ix += 1
    num_cards_left_in_shoe = len(shoe) - top_of_shoe_ix
    player_cardcount_signal = float( playerCardCounterUpdate(player_cardcount_signal,num_cards_left_in_shoe,nextCard))
    dealer_sum = abs(nextCard) #absolute value is taken for the dealer starting card

    if Verbose:
      print("==================")
      print(f"Player Starting Sum: {player_sum}")
      print(f"Dealer Starting Sum: {dealer_sum}")

    ### PLAYERS TURN ###
    if Verbose:
        print("--Player's Turn")

    player_is_active = playerStrategy(player_sum,dealer_sum,current_round_cardcount_signal) #Boolean flag for the Player still playing. True iff player wants to "hit" and keep going.
    player_is_active = player_is_active and ( top_of_shoe_ix < len(shoe) ) #Check and make sure there are still cards left in the shoe! Everything is automatically over if we are out of cards.
    player_is_busted = False #Boolean flag for the player being busted.



    while player_is_active: #while player wants to keep going
      #deal the next card and also let the player card counter see the next card
      nextCard = shoe[top_of_shoe_ix]
      top_of_shoe_ix += 1
      num_cards_left_in_shoe = len(shoe) - top_of_shoe_ix
      player_cardcount_signal = float(playerCardCounterUpdate(player_cardcount_signal,num_cards_left_in_shoe,nextCard))
      player_sum += nextCard

      if Verbose:
        print(f"{player_cardcount_signal=}")

      #check status for bustedness and what to do next
      player_is_busted = ( player_sum < 1 or player_sum > 21 ) #flag for busted
      player_is_active = ( not player_is_busted ) and playerStrategy(player_sum,dealer_sum,current_round_cardcount_signal) #check if player is still active
      player_is_active = player_is_active and ( top_of_shoe_ix < len(shoe) ) #if we are at out of cards in the shoe everything is automatically over!

      if Verbose:
        print(f"{player_sum = }, {player_is_busted=}, {player_is_active=}")

    ### DEALER'S TURN ###
    if Verbose:
        print("--Dealer's Turn")

    #The dealer will always hit if <=17 and player is not busted
    dealer_is_active = (not player_is_busted) #flag for the Dealer still playing. Note that if player busted, the dealer will just skip their turn.
    dealer_is_active = dealer_is_active and ( top_of_shoe_ix < len(shoe) ) #if we are out of cards everything is automatically over
    dealer_is_busted = False

    while dealer_is_active:
      #deal card to dealer
      nextCard = shoe[top_of_shoe_ix]
      top_of_shoe_ix += 1
      num_cards_left_in_shoe = len(shoe) - top_of_shoe_ix
      player_cardcount_signal = float(playerCardCounterUpdate(player_cardcount_signal,num_cards_left_in_shoe,nextCard))
      dealer_sum += nextCard

      #check for what to do
      dealer_is_busted = (dealer_sum < 1) or (dealer_sum > 21) #check if dealer busted
      dealer_is_active = (not dealer_is_busted) and ( dealer_sum <= 16 )#check if dealer chooses to hit and keep playing
      dealer_is_active = dealer_is_active and ( top_of_shoe_ix < len(shoe) ) #if we are out of cards everything is automatically over

      if Verbose:
        print(f"{dealer_sum = }, {dealer_is_busted=}, {dealer_is_active=}")

    player_wins = dealer_is_busted or ( (not player_is_busted) and player_sum > dealer_sum ) #Boolean variable if player wins!
    dealer_wins = player_is_busted or ( (not dealer_is_busted) and player_sum < dealer_sum ) #Boolean variable if dealer wins!
    if Verbose:
      print(f"{player_wins=}, {dealer_wins=}")

    player_bankroll += int( bet_size*player_wins - bet_size*dealer_wins ) #give player the money for winning or losing!

    #if you are over the table maximum move the excess to the savings account
    if player_bankroll >= TABLE_MAXIMUM_BET_SIZE:
      player_savings_account += (player_bankroll - TABLE_MAXIMUM_BET_SIZE)
      player_bankroll = TABLE_MAXIMUM_BET_SIZE

  #return the amount at the table plus in the savings account
  return int( player_bankroll + player_savings_account )