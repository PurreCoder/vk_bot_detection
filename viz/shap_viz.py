import shap
import numpy as np
import matplotlib.pyplot as plt
from shap import Explanation


class SHAPVisualizer:

    MAX_EXPLANATIONS = 15

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def plot_summary(self, shap_values, test_data):
        """Create SHAP summary plot with proper dimension handling"""
        print("Creating SHAP summary plot...")

        # Single array output
        plt.figure()

        shap.summary_plot(shap_values, test_data, feature_names=self.feature_names, max_display=len(self.feature_names), show=False)
        plt.title('SHAP Summary Plot', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('saves/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Summary plot saved successfully!")

    def plot_feature_importance(self, shap_values):
        """Create feature importance bar plot"""
        print("Creating feature importance plot...")

        try:
            overall_importance = np.mean(np.abs(shap_values), axis=0)

            # Sort features by importance
            sorted_idx = np.argsort(overall_importance)[::-1]
            sorted_features = [self.feature_names[i] for i in sorted_idx]
            sorted_importance = overall_importance[sorted_idx]

            plt.figure(figsize=(12, 8))
            plt.barh(range(len(sorted_importance[:15])), sorted_importance[:15])
            plt.yticks(range(len(sorted_importance[:15])), sorted_features[:15])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Top 15 Most Important Features (SHAP)')
            plt.gca().invert_yaxis()

            # Add value labels
            for i, v in enumerate(sorted_importance[:15]):
                plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

            plt.tight_layout()
            plt.savefig('saves/shap_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Feature importance plot saved successfully!")

        except Exception as e:
            print(f"❌ Feature importance plot failed: {e}")

    def plot_dependence(self, shap_values, test_data, feature_index):
        """Create dependence plot for a specific feature"""
        try:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(self.feature_names[feature_index], shap_values, test_data,
                                 feature_names=self.feature_names, show=False)
            plt.title(f'SHAP Dependence Plot - {self.feature_names[feature_index]}')
            plt.tight_layout()
            plt.savefig(f'saves/dependence_plots/{self.feature_names[feature_index]}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Dependence plot for {self.feature_names[feature_index]} saved!")

        except Exception as e:
            print(f"❌ Dependence plot failed: {e}")

    def plot_explanation(self, model_name, its_id, explanation):
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=len(self.feature_names), show=False)
        plt.title(f'SHAP Explanation for user id{its_id}')
        plt.tight_layout()
        plt.savefig(f'saves/explanation_plots/{model_name}_id{its_id}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_explanations(self, model_name, ids, explanations):
        explained_counter = 0
        for its_id, explanation in zip(ids, explanations):
            if np.any(explanation.values):
                if explained_counter >= SHAPVisualizer.MAX_EXPLANATIONS:
                    print(f'Provided explanations only for {SHAPVisualizer.MAX_EXPLANATIONS} users, randomly chosen out of {len(ids)} mispredictions')
                    break
                self.plot_explanation(model_name, its_id, explanation)
                explained_counter += 1
            else:
                print('Failed to explain prediction for user id', its_id)


    def create_comprehensive_report(self, shap_values, test_data):
        """Create comprehensive SHAP analysis"""

        print("\nCreating visualizations...")

        # Create summary plot
        self.plot_summary(shap_values, test_data)

        # Create feature importance plot
        self.plot_feature_importance(shap_values)

        # Plot dependence for top 10 features
        overall_importance = np.mean(np.abs(shap_values), axis=0)
        top_features_idx = np.argsort(overall_importance)[-10:][::-1]
        for idx in top_features_idx:
            self.plot_dependence(shap_values, test_data, idx)

        print("\n✅ SHAP analysis completed successfully!")
