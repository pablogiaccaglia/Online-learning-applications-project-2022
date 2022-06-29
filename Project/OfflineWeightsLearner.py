import numpy as np


class OfflineWeightsLearner:

    # the estimation of the probability can be computed node by node independently , so we need just the targetNodes

    @staticmethod
    def __generateEpisodesDataset(numOfEpisodes: int, probabilitiesMatrix) -> list[list[np.ndarray]]:

        def simulateEpisode(initialProbabilitiesMatrix: np.ndarray, numOfMaxSteps) -> list[np.ndarray]:
            probabilitiesMatrix = initialProbabilitiesMatrix.copy()
            numOfNodes = probabilitiesMatrix.shape[0]  # square matrix

            # active nodes at time 0 are chosen by drawing them from a binomial distribution with parameters 1, 0.1. it returns
            # an array of len numOfNodes, with values 0 or 1.

            initialActiveNodes = np.random.binomial(1, 0.5, size = numOfNodes)

            # the dataset to exploit to estimate probabilities,
            # whose first row is the initial active nodes just found
            history = np.array([initialActiveNodes])

            activeNodes = initialActiveNodes  # we store all the activated nodes in the episode
            newlyActiveNodes = activeNodes  # nodes activated in this time stamp. # e.g [0 0 1 0 0] for 5 nodes

            t = 0

            while t < numOfMaxSteps and np.sum(
                    newlyActiveNodes) > 0:  # loop until max number of steps is reached or no new nodes to activate exist
                p = (
                        probabilitiesMatrix.T * activeNodes).T  # we select from the probability matrix only the rows related to the active nodes.

                # p is a matrix of same dim of probabilitiesMatrix, with all zeros except the rows of the active nodes

                #  we compute the values of the activated edges by sampling a value from a random distribution, whose value
                # is from 0 and 1, and then compare this value with the probability of the corresponding edge: if the value of this drawn
                # probability is bigger than the value of the probability associated with the edge, then the edge is activated.
                # activatedEdges is a matrix of False and True values
                activatedEdges = p > np.random.rand(p.shape[0],
                                                    p.shape[1])  # one value for each edge of the active nodes!
                # it is a boolean matrix of dim NxN, where eventually the rows of the active nodes have True values if edge is activated

                # we remove from the probability matrix all the values of the probabilities related to the previously activated
                # nodes
                probabilitiesMatrix = probabilitiesMatrix * ((p != 0) == activatedEdges)  # ??? why??
                # matrix of dim NxN where the rows of the active nodes have value 0 if the corresponding edge has not been activated

                # now we compute the values of the newly activated nodes, the newly activated nodes are the ones having
                # an edge which has just been activated (we obviously exclude previously activated nodes)
                newlyActiveNodes = (np.sum(activatedEdges, axis = 0) > 0) * (
                        1 - activeNodes)  # nodes activated in this time stamp. # e.g [0 0 1 0 0] for 5 nodes

                activeNodes = np.array(activeNodes + newlyActiveNodes)
                history = np.concatenate((history, [newlyActiveNodes]), axis = 0)
                t = t + 1

            return history  # after the influence propagation ends, we return the history of activated edges

        dataset = []
        for e in range(0, numOfEpisodes):
            # we set 10 as number of max steps of the diffusion episode, even though usually the length of the diffusion episode is less than 10
            dataset.append(simulateEpisode(initialProbabilitiesMatrix = probabilitiesMatrix, numOfMaxSteps = 30))

        return dataset

    @staticmethod
    def estimateProbabilities(probabilitiesMatrix,
                              targetNodes,
                              numberOfNodes,
                              numOfEpisodes) -> np.ndarray:

        datasetOfDiffusionEpisodes = OfflineWeightsLearner.__generateEpisodesDataset(numOfEpisodes = numOfEpisodes,
                                                                                     probabilitiesMatrix = probabilitiesMatrix)

        estimatedProbs = np.empty((len(targetNodes), numberOfNodes))

        for index, node in enumerate(targetNodes):
            credits = np.zeros(numberOfNodes)  # here we store the credits assigned to each node in all episodes
            occurenciesVActive = np.zeros(numberOfNodes)  # here we store the occurences of each node in all episodes

            for episode in datasetOfDiffusionEpisodes:
                # first we localize the row (the time step of the episode) in which the node has been activated
                # episode is a matrix of dim N x N
                indexWAactive = np.argwhere(episode[:, node] == 1).reshape(
                        -1)  # reshape is due to the fact that from [1,1] we just need [1,]

                # now we check which nodes where active in the previous time step in which W was active, and assign them the credits
                if len(indexWAactive > 0) and indexWAactive[
                    0] > 0:  # we check indexWAactive[0] > 0 because we are going to look at previous time steps

                    # len(indexWAactive > 0) because we could have an episode in which we didn't finished the diffusion

                    # we are sure that in the previous time step at least 1 node has been activated, otherwise the cascade could not continue
                    activeNodesInPreviousTimestep = episode[indexWAactive[0] - 1, :]
                    # credits are assigned to the corresponding nodes in an uniform way
                    credits += activeNodesInPreviousTimestep / np.sum(
                            activeNodesInPreviousTimestep)  # we compute and update credits

                # now we have to check the occurencies of each node in the episode
                for v in range(0, numberOfNodes):
                    if v != node:
                        indexVActive = np.argwhere(episode[:, v] == 1).reshape(
                                -1)  # index of the time step in which node v was active

                        if len(indexVActive) > 0 and (
                                len(indexWAactive > 0) and indexVActive < indexWAactive or len(indexWAactive) == 0):
                            occurenciesVActive[v] += 1

            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                es = credits / occurenciesVActive
                estimatedProbs[index] = np.nan_to_num(es)

        return estimatedProbs
