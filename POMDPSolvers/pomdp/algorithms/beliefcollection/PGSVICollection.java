package pomdp.algorithms.beliefcollection;

import java.util.Iterator;
import java.util.Vector;
import java.util.Map.Entry;

import pomdp.algorithms.ValueIteration;
import pomdp.utilities.BeliefState;
import pomdp.utilities.ExecutionProperties;
import pomdp.utilities.Logger;

public class PGSVICollection extends BeliefCollection {

	int depth = 0;
	final int defaultMaxDepth = 200;
	double PICK_MDP_ACTION = 0.9;
	Vector<BeliefState> exploredBeliefs;

	public PGSVICollection(ValueIteration vi, boolean bAllowDuplicates) {
		super(vi, bAllowDuplicates);
	}

	@Override
	public Vector<BeliefState> expand(Vector<BeliefState> beliefPoints) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Vector<BeliefState> expand(int numNewBeliefs, Vector<BeliefState> beliefPoints) {
		exploredBeliefs = new Vector<BeliefState>();

		BeliefState bsInitial = POMDP.getBeliefStateFactory().getInitialBeliefState();

		do {
			/* pick an initial starting state */
			int iInitialState = -1;
			do {
				iInitialState = POMDP.chooseStartState();
			} while (POMDP.isTerminalState(iInitialState));

			double dDelta = weightedForwardSearch(iInitialState, bsInitial, 1, numNewBeliefs, numNewBeliefs);
		} while (exploredBeliefs.size() < numNewBeliefs);
		return exploredBeliefs;
	}

	protected double weightedForwardSearch(int iState, BeliefState bsCurrent, int iDepth, int maxDepth, int cMaxBeliefs) {
		double dDelta = 0.0, dNextDelta = 0.0;
		int iNextState = 0, iHeuristicAction = 0;
		Vector<Integer> iObservation;
		BeliefState bsNext = null;

		if (m_bAllowDuplicates || !exploredBeliefs.contains(bsCurrent))
			exploredBeliefs.add(bsCurrent);
		if (exploredBeliefs.size() == cMaxBeliefs)
			return 0.0;

		if ((POMDP.terminalStatesDefined() && POMDP.isTerminalState(iState)) || (iDepth == maxDepth)) {
			depth = iDepth;
			Logger.getInstance().logln(
					"Ended at depth " + iDepth + ". isTerminalState(" + iState + ")=" + POMDP.isTerminalState(iState));
		} else {
			iHeuristicAction = getAction(bsCurrent);
			iNextState = selectNextState(bsCurrent, iHeuristicAction);
			iObservation = getObservation(bsCurrent, iHeuristicAction);
			bsNext = getNextBeliefState(bsCurrent, iHeuristicAction, iObservation);

			if (bsNext == null || bsNext.equals((bsCurrent))) {
				// Logger.getInstance().logln( "Ended at depth " + iDepth + "
				// due to an error" );
				depth = iDepth;
			} else {
				dNextDelta = weightedForwardSearch(iNextState, bsNext, iDepth + 1, maxDepth, cMaxBeliefs);
			}
		}
		return Math.max(dDelta, dNextDelta);
	}

	private int getAction(BeliefState bs) {
		if (valueIteration.getRandomGenerator().nextDouble() < PICK_MDP_ACTION)
			return POMDP.getMDPValueFunction().getWeightedAction(bs);
		else
			return valueIteration.getRandomGenerator().nextInt(POMDP.getActionCount());
	}

	private Vector<Integer> getObservation(BeliefState bs, int iAction) {
		int iEndState = selectNextState(bs, iAction);
		Iterator<Entry<Integer, Double>> itNonZeroObservations = POMDP.getNonZeroObservations(iAction, iEndState);
		Vector<Integer> valuableObservation = new Vector<Integer>();
		Vector<Entry<Integer, Double>> vector = new Vector<Entry<Integer, Double>>();
		do {
			Entry<Integer, Double> e = itNonZeroObservations.next();
			vector.add(e);
		} while (itNonZeroObservations.hasNext());

		if (vector.size() <= 4) {
			for (int i = 0; i < vector.size(); i++) {
				valuableObservation.add(vector.get(i).getKey());
			}
		} else {
			for (int i = 0; i < vector.size(); i++) {
				if (POMDP.O(iAction, iEndState, vector.get(i).getKey()) >= 0.3)
					valuableObservation.add(vector.get(i).getKey());
			}
		}

		return valuableObservation;
	}

	private BeliefState getNextBeliefState(BeliefState bs, int iAction, Vector<Integer> iObservation) {
		BeliefState bsNext = null;
		double dMaxDist = 0.0, dDist = 0.0;
		for (int i = 0; i < iObservation.size(); i++) {
			if (bs.nextBeliefState(iAction, iObservation.get(i)) != null) {
				dDist = POMDP.getBeliefStateFactory().distance(exploredBeliefs,
						bs.nextBeliefState(iAction, iObservation.get(i)));
				if (dDist > dMaxDist) {
					dMaxDist = dDist;
					bsNext = bs.nextBeliefState(iAction, iObservation.get(i));
				}
			}
		}
		return bsNext;
	}

	private int selectNextState(BeliefState bs, int iAction) {
		int iBestNextState = 0;
		double iMaxProb = 0.0;
		for (int iNextState = 0; iNextState < POMDP.getStateCount(); iNextState++) {
			double iProb = 0.0;
			for (int iState = 0; iState < bs.countStates(); iState++) {
				iProb += POMDP.tr(iState, iAction, iNextState);
			}
			if (iProb > iMaxProb) {
				iMaxProb = iProb;
				iBestNextState = iNextState;
			}
		}
		return iBestNextState;
	}

	@Override
	public Vector<BeliefState> initialBelief() {
		/* initialize the MDP Heuristic */
		POMDP.getMDPValueFunction().valueIteration(1000, ExecutionProperties.getEpsilon());

		return new Vector<BeliefState>();
	}

}
