import numpy as np
from NSM.Agent.NSMAgent import NsmAgent
from NSM.Agent.TeacherAgent import TeacherAgent_hybrid
from NSM.Agent.TeacherAgent2 import TeacherAgent_parallel
from functools import reduce


def init_nsm(args, logger, num_entity, num_relation, num_word):
    logger.info("Building {}.".format("Agent"))
    # q_type = args['q_type']
    agent = NsmAgent(args, logger, num_entity, num_relation, num_word)
    # agent.model_def(instructor=instructor, reasoner=reasoner)
    logger.info("Architecture: {}".format(agent))
    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
                        for w in agent.parameters()])
    logger.info("Agent params: {}".format(total_params))

    return agent


def init_parallel(args, logger, num_entity, num_relation, num_word):
    logger.info("Building {}.".format("Agent"))
    agent = TeacherAgent_parallel(args, logger, num_entity, num_relation, num_word)
    logger.info("Architecture: {}".format(agent))
    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
                        for w in agent.parameters()])
    logger.info("Agent params: {}".format(total_params))

    return agent


def init_hybrid(args, logger, num_entity, num_relation, num_word):
    logger.info("Building {}.".format("Agent"))
    agent = TeacherAgent_hybrid(args, logger, num_entity, num_relation, num_word)
    logger.info("Architecture: {}".format(agent))
    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
                        for w in agent.parameters()])
    logger.info("Agent params: {}".format(total_params))

    return agent
