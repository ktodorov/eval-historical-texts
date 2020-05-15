import numpy as np

from copy import deepcopy
from collections import defaultdict, namedtuple

from enums.tag_metric import TagMetric
from enums.tag_measure_type import TagMeasureType

Entity = namedtuple("Entity", "e_type start_offset end_offset")


class TagMetricsService:
    """Calculates metrics related to tagging replicating CLEF scorer functions
    """

    def __init__(self):
        self.metrics = {
            TagMetric.Correct: 0,
            TagMetric.Incorrect: 0,
            TagMetric.Partial: 0,
            TagMetric.Missed: 0,
            TagMetric.Spurious: 0,
            TagMetric.Possible: 0,
            TagMetric.Actual: 0,
            TagMetric.TruePositives: 0,
            TagMetric.FalsePositives: 0,
            TagMetric.FalsePositives: 0,
            TagMetric.PrecisionMicro: 0,
            TagMetric.RecallMicro: 0,
            TagMetric.F1ScoreMicro: 0,
            TagMetric.PrecisionMacroDoc: [],
            TagMetric.RecallMacroDoc: [],
            TagMetric.F1ScoreMacroDoc: [],
            TagMetric.PrecisionMacro: 0,
            TagMetric.RecallMacro: 0,
            TagMetric.F1ScoreMacro: 0,
        }

        # four evaluation schemes
        self.metric_schema = {
            TagMeasureType.Strict: deepcopy(self.metrics),
            TagMeasureType.Partial: deepcopy(self.metrics),
        }

        self.reset()

    def reset(self):
        self._results = {}
        self._results_per_type = {}

    def initialize(self, entity_tag_type):
        # Create an accumulator to store overall results
        self._results[entity_tag_type] = deepcopy(self.metric_schema)
        self._results_per_type[entity_tag_type] = defaultdict(
            lambda: deepcopy(self.metric_schema))

    def calculate_batch(
            self,
            prediction_tags,
            target_tags,
            main_entities):

        results = deepcopy(self.metric_schema)
        results_per_type = defaultdict(lambda: deepcopy(self.metric_schema))

        results, results_per_type = self._aggregate_batch(
            results,
            results_per_type,
            prediction_tags,
            target_tags,
            main_entities)

        result = self._calculate_overall_stats(results, results_per_type)
        return result

    def _aggregate_batch(
            self,
            results,
            results_per_type,
            prediction_tags,
            target_tags,
            main_entities,
            calculate_doc_scores: bool = False):
        for (current_prediction_tags, current_target_tags) in zip(prediction_tags, target_tags):
            if calculate_doc_scores:
                doc_results = deepcopy(self.metric_schema)
                doc_results_per_type = defaultdict(
                    lambda: deepcopy(self.metric_schema))

            predicted_named_entities = self._get_named_entities(current_prediction_tags)
            true_named_entities = self._get_named_entities(current_target_tags)

            if len(predicted_named_entities) == 0 and len(true_named_entities) == 0:
                continue

            seg_results, seg_results_per_type = self._compute_metrics(
                true_named_entities, predicted_named_entities, main_entities)

            # accumulate overall stats
            results, results_per_type = self._accumulate_stats(
                results, results_per_type, seg_results, seg_results_per_type)

            if calculate_doc_scores:
                 # accumulate stats across documents
                doc_results, doc_results_per_type = self._accumulate_stats(
                    doc_results, doc_results_per_type, seg_results, seg_results_per_type)

            for e_type in results_per_type:
                if calculate_doc_scores:
                    doc_results_per_type[e_type] = self._compute_precision_recall_wrapper(
                        doc_results_per_type[e_type])

                    results_per_type[e_type] = self._accumulate_doc_scores(
                        results_per_type[e_type], doc_results_per_type[e_type]
                    )

            if calculate_doc_scores:
                doc_results = self._compute_precision_recall_wrapper(doc_results)
                results = self._accumulate_doc_scores(results, doc_results)

        # results = self._compute_precision_recall_wrapper(results)
        # for e_type in results_per_type:
        #     results_per_type[e_type] = self._compute_precision_recall_wrapper(
        #         results_per_type[e_type])

        return results, results_per_type

    def _accumulate_doc_scores(self, results, doc_results):
        """Accumulate the scores (P, R, F1) across documents.

        When a entity does not occur in a particular document according to the gold standard,
        it is dismissed as it would artifically lower the final measure.

        :param dict results: nested accumulator of scores across document.
        :param dict doc_results: nested scores of current document.
        :return: accumulator updated with the scores of current document.
        :rtype: dict

        """

        for eval_schema in results:
            actual = doc_results[eval_schema][TagMetric.Actual]
            possible = doc_results[eval_schema][TagMetric.Possible]

            # to compute precision dismiss documents for which no entities were predicted
            if actual != 0:
                results[eval_schema][TagMetric.PrecisionMacroDoc].append(
                    doc_results[eval_schema][TagMetric.PrecisionMicro])

            # to compute recall dismiss documents for which no entities exists in gold standard
            if possible != 0:
                results[eval_schema][TagMetric.RecallMacroDoc].append(
                    doc_results[eval_schema][TagMetric.RecallMicro])

            # to compute recall dismiss documents for which no entities exists in gold standard
            if possible != 0 and actual != 0:
                results[eval_schema][TagMetric.F1ScoreMacroDoc].append(
                    doc_results[eval_schema][TagMetric.F1ScoreMicro])

        return results

    def add_predictions(
            self,
            prediction_tags,
            target_tags,
            main_entities,
            entity_tag_type):
        if entity_tag_type not in self._results.keys():
            self.initialize(entity_tag_type)

        self._results[entity_tag_type], self._results_per_type[entity_tag_type] = self._aggregate_batch(
            self._results[entity_tag_type],
            self._results_per_type[entity_tag_type],
            prediction_tags,
            target_tags,
            main_entities,
            calculate_doc_scores=True)

    def calculate_overall_stats(self):
        result = {}
        for entity_tag_type in self._results.keys():
            result[entity_tag_type] = self._calculate_overall_stats(
                self._results[entity_tag_type], self._results_per_type[entity_tag_type])

        return result

    def _calculate_overall_stats(self, results, results_per_type):
        # Compute overall metrics by entity type
        for e_type in results_per_type:
            results_per_type[e_type] = self._compute_precision_recall_wrapper(results_per_type[e_type])
            results_per_type[e_type] = self._compute_macro_doc_scores(results_per_type[e_type])

        # Compute overall metrics across entity types
        results = self._compute_precision_recall_wrapper(results)
        results = self._compute_macro_doc_scores(results)
        results = self._compute_macro_type_scores(results, results_per_type)

        return results, results_per_type


    def _compute_macro_doc_scores(self, results):
        """Compute the macro scores for Precision, Recall, F1 across documents.

        The score is a simple average across documents.

        :param dict results: evaluation results.
        :return: updated evaluation results.
        :rtype: dict

        """

        metrics = (TagMetric.PrecisionMacroDoc, TagMetric.RecallMacroDoc, TagMetric.F1ScoreMacroDoc)

        for eval_schema in results:
            for metric in metrics:
                vals = results[eval_schema][metric]
                results[eval_schema][metric] = float(np.mean(vals)) if len(vals) > 0 else 0

                std_metric = metric.value + "-std"
                tag_metric = TagMetric(std_metric)
                results[eval_schema][tag_metric] = float(np.std(vals)) if len(vals) > 0 else 0

        return results


    def _compute_macro_type_scores(self, results, results_per_type):
        """Compute the macro scores for Precision, Recall, F1 across entity types.


        There are different ways to comput the macro F1-scores across class.
        Please see the explanations at:
        https://towardsdatascience.com/a-tale-of-two-macro-f1s-8811ddcf8f04

        :param dict results: evaluation results.
        :param dict results_per_type: evaluation results per type.
        :return: updated results and results per type.
        :rtype: Tuple(dict, dict)

        """

        for eval_schema in results:
            precision_sum = 0
            recall_sum = 0
            f1_sum = 0

            n_tags = len(results_per_type)
            if n_tags == 0:
                continue

            for tag in results_per_type:
                precision_sum += results_per_type[tag][eval_schema][TagMetric.PrecisionMicro]
                recall_sum += results_per_type[tag][eval_schema][TagMetric.RecallMicro]
                f1_sum += results_per_type[tag][eval_schema][TagMetric.F1ScoreMicro]

            precision_macro = precision_sum / n_tags
            recall_macro = recall_sum / n_tags
            f1_macro_mean = f1_sum / n_tags
            f1_macro_recomp = (
                2 * (precision_macro * recall_macro) /
                (precision_macro + recall_macro)
                if (precision_macro + recall_macro) > 0
                else 0
            )

            results[eval_schema][TagMetric.PrecisionMacro] = precision_macro
            results[eval_schema][TagMetric.RecallMacro] = recall_macro
            # sklearn-style
            results[eval_schema][TagMetric.F1ScoreMacro] = f1_macro_mean
            results[eval_schema][TagMetric.F1ScoreMacroRecomputed] = f1_macro_recomp

        return results

    def _accumulate_stats(self, results, results_per_type, tmp_results, tmp_results_per_type):
        """Accumulate the scores across lines.

        :param dict results: nested accumulator of scores across lines.
        :param dict results_per_type: nested accumulator of scores per type across lines.
        :param dict tmp_results: scores of current line.
        :param dict tmp_results_per_type: scores of current line per type.
        :return: updated accumulator across labels and per entity type.
        :rtype: Tuple(dict, dict)

        """

        for eval_schema in results:
            # Aggregate metrics across entity types
            for metric in results[eval_schema]:
                results[eval_schema][metric] += tmp_results[eval_schema][metric]

                # Aggregate metrics by entity type
                for e_type in tmp_results_per_type:
                    results_per_type[e_type][eval_schema][metric] += tmp_results_per_type[e_type][
                        eval_schema
                    ][metric]

        return results, results_per_type

    def _compute_precision_recall_wrapper(self, results):
        """
        Wraps the compute_precision_recall function and runs it for each evaluation scenario in results
        """

        results_a = {
            key: self._compute_precision_recall(value, True)
            for key, value in results.items()
            if key in [TagMeasureType.Partial]
        }

        # in the entity type matching scenario (fuzzy),
        # overlapping entities and entities with strict boundary matches are rewarded equally
        results_b = {
            key: self._compute_precision_recall(value)
            for key, value in results.items()
            if key in [TagMeasureType.Strict]
        }

        results = {**results_a, **results_b}

        return results

    def _compute_precision_recall(self, results, partial=False):
        """ Compute the micro scores for Precision, Recall, F1.

        :param dict results: evaluation results.
        :param bool partial: option to half the reward of partial matches.
        :return: Description of returned object.
        :rtype: updated results

        """

        actual = results[TagMetric.Actual]
        possible = results[TagMetric.Possible]
        partial = results[TagMetric.Partial]
        correct = results[TagMetric.Correct]

        if partial:
            precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
            recall = (correct + 0.5 * partial) / \
                possible if possible > 0 else 0

        else:
            precision = correct / actual if actual > 0 else 0
            recall = correct / possible if possible > 0 else 0

        results[TagMetric.PrecisionMicro] = precision
        results[TagMetric.RecallMicro] = recall
        results[TagMetric.F1ScoreMicro] = (
            2 * (precision * recall) / (precision +
                                        recall) if (precision + recall) > 0 else 0
        )

        return results

    def _get_named_entities(self, prediction_tags):
        named_entities = []
        start_offset = None
        end_offset = None
        ent_type = None

        for offset, token_tag in enumerate(prediction_tags):
            if token_tag == "O":
                if ent_type is not None and start_offset is not None:
                    end_offset = offset - 1
                    named_entities.append(
                        Entity(ent_type, start_offset, end_offset))
                    start_offset = None
                    end_offset = None
                    ent_type = None

            elif ent_type is None:
                ent_type = token_tag[2:]
                start_offset = offset

            elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == "B"):

                end_offset = offset - 1
                named_entities.append(
                    Entity(ent_type, start_offset, end_offset))

                # start of a new entity
                ent_type = token_tag[2:]
                start_offset = offset
                end_offset = None

        # catches an entity that goes up until the last token
        if ent_type and start_offset and end_offset is None:
            named_entities.append(
                Entity(ent_type, start_offset, len(prediction_tags) - 1))

        # align shape of NE and link objects as the latter allows alternative annotations
        named_entities = [[ne] for ne in named_entities]

        return named_entities

    def _compute_metrics(self, true_named_entities: list, pred_named_entities: list, tags: set):
        """Compute the metrics of segment for all evaluation scenarios.

        :param list(Entity) true_named_entities: nested list with entity annotations of gold standard.
        :param list(Entity) pred_named_entities: nested list with entity annotations of system response.
        :param set tags: limit to provided tags.
        :return: nested results and results per entity type
        :rtype: Tuple(dict, dict)

        """

        # overall results
        evaluation = deepcopy(self.metric_schema)

        # results by entity type
        evaluation_agg_entities_type = defaultdict(
            lambda: deepcopy(self.metric_schema))

        # keep track of entities that overlapped
        true_which_overlapped_with_pred = []

        # Subset into only the tags that we are interested in.
        # NOTE: we remove the tags we don't want from both the predicted and the
        # true entities. This covers the two cases where mismatches can occur:
        #
        # 1) Where the model predicts a tag that is not present in the true data
        # 2) Where there is a tag in the true data that the model is not capable of
        # predicting.

        # only allow alternatives in prediction file, not in gold standard
        true_named_entities = [ent[0]
                               for ent in true_named_entities if ent[0].e_type in tags]
        # pred_named_entities = [ent for ent in pred_named_entities if [ent[0]].e_type in tags]
        pred_named_entities = [ent for ent in pred_named_entities if any(
            [e.e_type in tags for e in ent])]

        # go through each predicted named-entity
        for pred in pred_named_entities:
            found_overlap = False

            # Check each of the potential scenarios in turn. See
            # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
            # for scenario explanation.

            # Scenario I: Exact match between true and pred
            for true_tag in true_named_entities:
                if any(p == true_tag for p in pred):
                    true_which_overlapped_with_pred.append(true_tag)
                    evaluation[TagMeasureType.Strict][TagMetric.Correct] += 1
                    evaluation[TagMeasureType.Partial][TagMetric.Correct] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[true_tag.e_type][TagMeasureType.Strict][TagMetric.Correct] += 1
                    evaluation_agg_entities_type[true_tag.e_type][TagMeasureType.Partial][TagMetric.Correct] += 1

                    break

            else:

                # check for overlaps with any of the true entities
                for true_tag in true_named_entities:

                    # NOTE: error in original code: missing + 1
                    # overlapping needs to take into account last token as well
                    pred_range = range(
                        pred[0].start_offset, pred[0].end_offset + 1)
                    true_range = range(true_tag.start_offset,
                                       true_tag.end_offset + 1)

                    # Scenario IV: Offsets match, but entity type is wrong

                    if (
                        true_tag.start_offset == pred[0].start_offset
                        and pred[0].end_offset == true_tag.end_offset
                        and true_tag.e_type != pred[0].e_type
                    ):

                        # overall results
                        evaluation[TagMeasureType.Strict][TagMetric.Incorrect] += 1
                        evaluation[TagMeasureType.Partial][TagMetric.Correct] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[true_tag.e_type][TagMeasureType.Strict][TagMetric.Incorrect] += 1
                        evaluation_agg_entities_type[true_tag.e_type][TagMeasureType.Partial][TagMetric.Correct] += 1

                        true_which_overlapped_with_pred.append(true_tag)
                        found_overlap = True

                        break

                    # check for an overlap, i.e. not exact boundary match, with true entities
                    # NOTE: error in original code:
                    # overlaps with true entities must only counted once
                    elif (
                        self._find_overlap(true_range, pred_range)
                        and true_tag not in true_which_overlapped_with_pred
                    ):

                        true_which_overlapped_with_pred.append(true_tag)
                        found_overlap = True

                        # Scenario V: There is an overlap (but offsets do not match
                        # exactly), and the entity type is the same.
                        # 2.1 overlaps with the same entity type

                        if any(p.e_type == true_tag.e_type for p in pred):

                            # overall results
                            evaluation[TagMeasureType.Strict][TagMetric.Incorrect] += 1
                            evaluation[TagMeasureType.Partial][TagMetric.Partial] += 1

                            # aggregated by entity type results
                            evaluation_agg_entities_type[true_tag.e_type][TagMeasureType.Strict][TagMetric.Incorrect] += 1
                            evaluation_agg_entities_type[true_tag.e_type][TagMeasureType.Partial][TagMetric.Partial] += 1

                            break

                        # Scenario VI: Entities overlap, but the entity type is
                        # different.

                        else:
                            # overall results
                            evaluation[TagMeasureType.Strict][TagMetric.Incorrect] += 1
                            evaluation[TagMeasureType.Partial][TagMetric.Partial] += 1

                            # aggregated by entity type results
                            # Results against the true entity

                            evaluation_agg_entities_type[true_tag.e_type][TagMeasureType.Strict][TagMetric.Incorrect] += 1
                            evaluation_agg_entities_type[true_tag.e_type][TagMeasureType.Partial][TagMetric.Partial] += 1

                            break

                # Scenario II: Entities are spurious (i.e., over-generated).

                if not found_overlap:

                    # Overall results
                    evaluation[TagMeasureType.Strict][TagMetric.Spurious] += 1
                    evaluation[TagMeasureType.Partial][TagMetric.Spurious] += 1

                    # Aggregated by entity type results

                    # NOTE: error in original code:
                    # a spurious entity for a particular tag should be only
                    # attributed to the respective tag
                    if pred[0].e_type in tags:
                        spurious_tags = [pred[0].e_type]

                    else:
                        # NOTE: when pred.e_type is not found in tags
                        # or when it simply does not appear in the test set, then it is
                        # spurious, but it is not clear where to assign it at the tag
                        # level. In this case, it is applied to all target_tags
                        # found in this example. This will mean that the sum of the
                        # evaluation_agg_entities will not equal evaluation.

                        spurious_tags = tags

                    for true_tag in spurious_tags:
                        evaluation_agg_entities_type[true_tag][TagMeasureType.Strict][TagMetric.Spurious] += 1
                        evaluation_agg_entities_type[true_tag][TagMeasureType.Partial][TagMetric.Spurious] += 1

        # Scenario III: Entity was missed entirely.

        for true_tag in true_named_entities:
            if true_tag not in true_which_overlapped_with_pred:
                # overall results
                evaluation[TagMeasureType.Strict][TagMetric.Missed] += 1
                evaluation[TagMeasureType.Partial][TagMetric.Missed] += 1

                # aggregated by entity type results
                evaluation_agg_entities_type[true_tag.e_type][TagMeasureType.Strict][TagMetric.Missed] += 1
                evaluation_agg_entities_type[true_tag.e_type][TagMeasureType.Partial][TagMetric.Missed] += 1

        # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
        # overall results, and use these to calculate precision and recall.
        for eval_type in evaluation:
            # if eval_type != "slot_error_rate":
            evaluation[eval_type] = self._compute_actual_possible(
                evaluation[eval_type])

        # Compute 'possible', 'actual', and precision and recall on entity level
        # results. Start by cycling through the accumulated results.
        for entity_type, entity_level in evaluation_agg_entities_type.items():

            # Cycle through the evaluation types for each dict containing entity
            # level results.
            for eval_type in entity_level:
                # if eval_type != "slot_error_rate":
                evaluation_agg_entities_type[entity_type][eval_type] = self._compute_actual_possible(
                    entity_level[eval_type]
                )

        return evaluation, evaluation_agg_entities_type

    def _find_overlap(self, true_range, pred_range):
        """Find the overlap between two ranges

        Find the overlap between two ranges. Return the overlapping values if
        present, else return an empty set().

        Examples:

        >>> _find_overlap((1, 2), (2, 3))
        2
        >>> _find_overlap((1, 2), (3, 4))
        set()
        """

        true_set = set(true_range)
        pred_set = set(pred_range)

        overlaps = true_set.intersection(pred_set)

        return overlaps

    def _compute_actual_possible(self, results):
        """Update the counts of possible and actual based on evaluation.

        :param dict results: results with updated evaluation counts.
        :return: the results dict with actual, possible updated.
        :rtype: dict

        """

        correct = results[TagMetric.Correct]
        incorrect = results[TagMetric.Incorrect]
        partial = results[TagMetric.Partial]
        missed = results[TagMetric.Missed]
        spurious = results[TagMetric.Spurious]

        # Possible: number annotations in the gold-standard which contribute to the
        # final score
        possible = correct + incorrect + partial + missed

        # Actual: number of annotations produced by the NER system
        actual = correct + incorrect + partial + spurious

        results[TagMetric.Actual] = actual
        results[TagMetric.Possible] = possible

        results[TagMetric.TruePositives] = correct
        results[TagMetric.FalsePositives] = actual - correct
        results[TagMetric.FalseNegatives] = possible - correct

        return results
