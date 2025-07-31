def create_visualizations(self, use_advanced_viz: bool = True, 
						visualization_mode: str = 'comprehensive',
						plot_style: str = 'binned_statistical'):
	"""Generate comprehensive publication-quality visualizations from parameter sweep results.
	
	Args:
		use_advanced_viz: Whether to use publication-quality styling (always True now)
		visualization_mode: Mode for visualization ('basic', 'comprehensive', 'publication')
			- 'comprehensive': Always generates all 4 visualization styles regardless of plot_style
			- 'basic' or 'publication': Uses specified plot_style only
		plot_style: Visualization approach ('binned_statistical', 'boxplot', 'individual_trajectories', 'violin', 'all')
			- Ignored when visualization_mode='comprehensive' (all styles generated)
	"""
	if not self.results:
		self.console.print("[yellow]No results to visualize[/yellow]")
		return
	
	# Create visualization directory
	viz_dir = self.output_dir / 'visualizations'
	viz_dir.mkdir(exist_ok=True)
	
	self.console.print(f"[blue]Creating publication-quality visualizations with '{plot_style}' style...[/blue]")
	
	# Setup publication-quality styling
	has_pub_style = self._setup_publication_style()
	if has_pub_style:
		self.console.print("[green]Using LearnM8 publication styling system[/green]")
	else:
		self.console.print("[yellow]Using fallback styling (LearnM8 visualization package not available)[/yellow]")
	
	try:
		# Load and prepare cycle-by-cycle data
		cycle_df = self._load_and_prepare_cycle_data()
		self.console.print(f"[green]Loaded {len(cycle_df)} cycle data points from {len(cycle_df['experiment_id'].unique())} experiments[/green]")
		
		# Generate plots with specified style(s)
		if plot_style == 'all' or visualization_mode == 'comprehensive':
			# Generate all four visualization styles for comprehensive mode
			if visualization_mode == 'comprehensive':
				self.console.print("[blue]Comprehensive mode: generating all 4 visualization styles...[/blue]")
			styles = ['binned_statistical', 'boxplot', 'individual_trajectories', 'violin']
			for style in styles:
				self.console.print(f"[blue]  • Generating {style} visualizations...[/blue]")
				self._create_comprehensive_plot_suite(cycle_df, viz_dir, style)
			self.console.print(f"[green]Generated visualizations in all 4 styles ({', '.join(styles)})[/green]")
		else:
			# Generate single style
			self._create_comprehensive_plot_suite(cycle_df, viz_dir, plot_style)
		
		self.console.print(f"[green]All visualizations completed and saved to {viz_dir}[/green]")
		
	except FileNotFoundError as e:
		self.console.print(f"[red]Data file not found: {e}[/red]")
		self.console.print("[yellow]Make sure to run parameter sweep with cycle-by-cycle data export enabled[/yellow]")
	except Exception as e:
		self.console.print(f"[red]Error creating visualizations: {e}[/red]")
		import traceback
		traceback.print_exc()

def _create_comprehensive_plot_suite(self, cycle_df: pd.DataFrame, viz_dir: Path, plot_style: str = 'binned_statistical'):
	"""Create the complete suite of requested publication-quality plots.
	
	Args:
		cycle_df: DataFrame with cycle-by-cycle data
		viz_dir: Directory to save visualizations
		plot_style: Visualization style ('binned_statistical', 'boxplot', 'individual_trajectories', 'violin')
	"""
	plot_count = 0
	
	try:
		# A) Top K recovery vs % explored
		self._create_top_k_recovery_plots(cycle_df, viz_dir, plot_style)
		plot_count += 3  # 3 versions (by model, strategy, initial)
		
		# B) EF vs % explored 
		self._create_ef_vs_explored_plots(cycle_df, viz_dir, plot_style)
		plot_count += 3
		
		# C-G) Performance metrics vs % explored
		self._create_performance_vs_explored_plots(cycle_df, viz_dir, plot_style)
		plot_count += 15  # 5 metrics × 3 versions each
		
		# H) Average score vs % explored
		self._create_avg_score_vs_explored_plots(cycle_df, viz_dir, plot_style)
		plot_count += 3
		
		# I) Intra-batch diversity vs % explored
		self._create_diversity_vs_explored_plots(cycle_df, viz_dir, plot_style)
		plot_count += 3
		
		# J) Parameter combination performance matrix (always uses heatmap style)
		self._create_parameter_matrix_plot(cycle_df, viz_dir)
		plot_count += 4  # 4 different metrics
		
		# K) Uncertainty vs performance trajectories (always uses trajectory style)
		self._create_uncertainty_performance_trajectories(cycle_df, viz_dir)
		plot_count += 3  # 3 different performance metrics
		
		# L) Uncertainty vs % explored
		self._create_uncertainty_vs_explored_plots(cycle_df, viz_dir, plot_style)
		plot_count += 3
		
		self.console.print(f"[green]Generated approximately {plot_count} publication-quality visualizations using '{plot_style}' style[/green]")
		
	except Exception as e:
		self.console.print(f"[red]Error in plot suite generation: {e}[/red]")
		raise

def _setup_publication_style(self):
	"""Setup publication-quality plotting style using LearnM8 standards."""
	try:
		from learnm8.visualization.publication_style import PublicationStyle
		self.pub_style = PublicationStyle()
		self.pub_style.apply_journal_style('nature')
		return True
	except (ImportError, TypeError, AttributeError):
		# Fallback to basic styling
		plt.style.use('default')  # Use default instead of seaborn
		sns.set_palette("colorblind")
		plt.rcParams.update({
			'font.size': 12,
			'axes.labelsize': 14,
			'axes.titlesize': 16,
			'xtick.labelsize': 12,
			'ytick.labelsize': 12,
			'legend.fontsize': 12,
			'figure.titlesize': 18,
			'figure.dpi': 300
		})
		return False

def _load_and_prepare_cycle_data(self):
	"""Load cycle-by-cycle data and calculate % explored."""
	cycle_csv_file = self.output_dir / 'parameter_sweep_cycle_by_cycle.csv'
	if not cycle_csv_file.exists():
		raise FileNotFoundError(f"Cycle-by-cycle data not found: {cycle_csv_file}")
	
	cycle_df = pd.read_csv(cycle_csv_file)
	if cycle_df.empty:
		raise ValueError("Cycle-by-cycle data is empty")
	
	# Calculate % explored
	cycle_df['percent_explored'] = (cycle_df['cumulative_labeled_mean'] / cycle_df['original_pool_size_mean']) * 100
	
	return cycle_df

# Method 1: Binned Statistical Trajectories (Recommended Primary Approach)
def _create_binned_statistical_plot(self, cycle_df: pd.DataFrame, metric_col: str, group_col: str,
									title: str, ylabel: str, viz_dir: Path, filename: str,
									subplot_configs: list = None, ground_truth_col: str = None,
									n_bins: int = 25, confidence_level: float = 0.95):
	"""Create clean trajectory plots using cycle-based aggregation with confidence intervals.
	
	This eliminates chaotic lines by aggregating data by cycle number and showing statistical summaries.
	"""
	try:
		import numpy as np
		from scipy import stats
		
		if subplot_configs:
			# Multi-subplot plot
			n_subplots = len(subplot_configs)
			cols = min(3, n_subplots)
			rows = (n_subplots + cols - 1) // cols
			
			fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
			if n_subplots == 1:
				axes = [axes]
			elif rows == 1:
				axes = axes if hasattr(axes, '__len__') else [axes]
			else:
				axes = axes.flatten()
			
			for i, config in enumerate(subplot_configs):
				if i >= len(axes):
					break
				
				ax = axes[i]
				metric = config['metric']
				
				# Strategy-agnostic plotting: use percent_explored for custom cycles, cycle numbers for traditional
				for group in sorted(cycle_df[group_col].unique()):
					group_data = cycle_df[cycle_df[group_col] == group]
					if group_data.empty or metric not in group_data.columns:
						continue
					
					# Determine x-axis variable based on experiment type
					x_var = self._determine_x_axis_variable(group_data)
					
					# Group by x-axis variable and calculate statistics
					if x_var == 'percent_explored':
						# For custom cycles: bin by percent explored to create smooth trajectories
						group_data = group_data.sort_values('percent_explored')
						bins = np.linspace(group_data['percent_explored'].min(), 
											group_data['percent_explored'].max(), 
											min(10, len(group_data)))
						group_data['x_binned'] = pd.cut(group_data['percent_explored'], bins=bins, include_lowest=True)
						cycle_stats = group_data.groupby('x_binned')[metric].agg(['mean', 'std', 'count']).reset_index()
						# Use bin centers for x-axis
						cycle_stats['x_axis'] = cycle_stats['x_binned'].apply(lambda x: x.mid)
					else:
						# Traditional experiments: use cycle numbers
						cycle_stats = group_data.groupby('cycle')[metric].agg(['mean', 'std', 'count']).reset_index()
						cycle_stats['x_axis'] = cycle_stats['cycle']
					
					cycle_stats = cycle_stats.dropna()
					
					if len(cycle_stats) < 2:
						continue
					
					# Calculate confidence intervals using t-distribution
					alpha = 1 - confidence_level
					cycle_stats['sem'] = cycle_stats['std'] / np.sqrt(cycle_stats['count'])
					# Handle single observations (count=1) by using std directly
					single_obs = cycle_stats['count'] == 1
					cycle_stats.loc[single_obs, 'sem'] = cycle_stats.loc[single_obs, 'std']
					t_crit = stats.t.ppf(1 - alpha/2, np.maximum(cycle_stats['count'] - 1, 1))
					cycle_stats['ci_lower'] = cycle_stats['mean'] - t_crit * cycle_stats['sem']
					cycle_stats['ci_upper'] = cycle_stats['mean'] + t_crit * cycle_stats['sem']
					
					# Clean trajectory plot using appropriate x-axis
					ax.plot(cycle_stats['x_axis'], cycle_stats['mean'], 
							label=group, marker='o', markersize=5, linewidth=2.5, alpha=0.9)
					
					# Confidence interval band (no legend entry for cleaner legend)
					ax.fill_between(cycle_stats['x_axis'], cycle_stats['ci_lower'], cycle_stats['ci_upper'],
									alpha=0.2)
				
				# Add ground truth line if specified (no legend entry for cleaner legend)
				if config.get('ground_truth_col') and config['ground_truth_col'] in cycle_df.columns:
					gt_value = cycle_df[config['ground_truth_col']].iloc[0]
					if not pd.isna(gt_value):
						ax.axhline(y=gt_value, color='red', linestyle='--', alpha=0.8, linewidth=2)
						# Add text annotation instead of legend entry
						ax.text(0.02, 0.98, f'Ground Truth = {gt_value:.2f}', 
								transform=ax.transAxes, verticalalignment='top',
								bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
								fontsize=8, color='red')
				
				# Set appropriate x-axis label based on data type
				x_label = 'Cycle Number' if x_var == 'cycle' else 'Percent Explored (%)'
				ax.set_xlabel(x_label)
				ax.set_ylabel(config['ylabel'])
				ax.set_title(config['title'])
				ax.grid(True, alpha=0.3)
				
				# Compact legend configuration for EF plots
				if 'EF' in config.get('title', ''):
					ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
								fontsize=8, frameon=True, fancybox=False, 
								shadow=False, framealpha=0.9, edgecolor='gray',
								columnspacing=0.5, handletextpad=0.3)
				else:
					ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
			
			# Hide unused subplots
			for i in range(n_subplots, len(axes)):
				axes[i].set_visible(False)
			
		else:
			# Single plot
			fig, ax = plt.subplots(figsize=(12, 8))
			
			# Use cycle numbers directly instead of binning
			for group in sorted(cycle_df[group_col].unique()):
				group_data = cycle_df[cycle_df[group_col] == group]
				if group_data.empty or metric_col not in group_data.columns:
					continue
				
				# Group by cycle number and calculate statistics
				cycle_stats = group_data.groupby('cycle')[metric_col].agg(['mean', 'std', 'count']).reset_index()
				cycle_stats = cycle_stats.dropna()
				
				if len(cycle_stats) < 2:
					continue
				
				# Calculate confidence intervals using t-distribution
				alpha = 1 - confidence_level
				cycle_stats['sem'] = cycle_stats['std'] / np.sqrt(cycle_stats['count'])
				# Handle single observations (count=1) by using std directly
				single_obs = cycle_stats['count'] == 1
				cycle_stats.loc[single_obs, 'sem'] = cycle_stats.loc[single_obs, 'std']
				t_crit = stats.t.ppf(1 - alpha/2, np.maximum(cycle_stats['count'] - 1, 1))
				cycle_stats['ci_lower'] = cycle_stats['mean'] - t_crit * cycle_stats['sem']
				cycle_stats['ci_upper'] = cycle_stats['mean'] + t_crit * cycle_stats['sem']
				
				# Clean trajectory plot by cycle
				ax.plot(cycle_stats['cycle'], cycle_stats['mean'], 
						label=group, marker='o', markersize=5, linewidth=2.5, alpha=0.9)
				
				# Confidence interval band
				ax.fill_between(cycle_stats['cycle'], cycle_stats['ci_lower'], cycle_stats['ci_upper'],
								alpha=0.2)
			
			# Add ground truth line if specified (no legend entry for cleaner legend)
			if ground_truth_col and ground_truth_col in cycle_df.columns:
				gt_value = cycle_df[ground_truth_col].iloc[0]
				if not pd.isna(gt_value):
					ax.axhline(y=gt_value, color='red', linestyle='--', alpha=0.8, linewidth=2)
					# Add text annotation instead of legend entry
					ax.text(0.02, 0.98, f'Ground Truth = {gt_value:.2f}', 
							transform=ax.transAxes, verticalalignment='top',
							bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
							fontsize=8, color='red')
			
			ax.set_xlabel('Cycle Number')
			ax.set_ylabel(ylabel)
			ax.set_title(title)
			ax.grid(True, alpha=0.3)
			
			# Compact legend configuration for EF plots
			if 'EF' in title or 'Enrichment Factor' in title:
				ax.legend(fontsize=8, frameon=True, fancybox=False, 
							shadow=False, framealpha=0.9, edgecolor='gray',
							columnspacing=0.5, handletextpad=0.3, 
							bbox_to_anchor=(1.02, 1), loc='upper left')
			else:
				ax.legend()
		
		plt.tight_layout()
		plt.savefig(viz_dir / filename, dpi=300, bbox_inches='tight')
		plt.close()
		
	except Exception as e:
		self.console.print(f"[yellow]Warning: Could not create binned statistical plot {filename}: {e}[/yellow]")
		import traceback
		traceback.print_exc()

# Method 2: Box Plots at Discrete Intervals
def _create_boxplot_intervals_plot(self, cycle_df: pd.DataFrame, metric_col: str, group_col: str,
									title: str, ylabel: str, viz_dir: Path, filename: str,
									n_intervals: int = 5, subplot_configs: list = None):
	"""Create box plots showing distributions at discrete % explored intervals."""
	try:
		import numpy as np
		
		if subplot_configs:
			# Multi-subplot plot
			n_subplots = len(subplot_configs)
			cols = min(3, n_subplots)
			rows = (n_subplots + cols - 1) // cols
			
			fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
			if n_subplots == 1:
				axes = [axes]
			elif rows == 1:
				axes = axes if hasattr(axes, '__len__') else [axes]
			else:
				axes = axes.flatten()
			
			for i, config in enumerate(subplot_configs):
				if i >= len(axes):
					break
				
				ax = axes[i]
				metric = config['metric']
				
				# Create discrete intervals
				percent_min = cycle_df['percent_explored'].min()
				percent_max = cycle_df['percent_explored'].max()
				bins = np.linspace(percent_min, percent_max, n_intervals + 1)
				labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}%' for i in range(n_intervals)]
				
				# Add interval categories
				cycle_df_copy = cycle_df.copy()
				cycle_df_copy['interval'] = pd.cut(cycle_df_copy['percent_explored'], bins=bins, labels=labels)
				
				# Box plot
				sns.boxplot(data=cycle_df_copy, x='interval', y=metric, hue=group_col, ax=ax, palette='Set2')
				ax.set_xlabel('% Explored Range')
				ax.set_ylabel(config['ylabel'])
				ax.set_title(config['title'])
				ax.tick_params(axis='x', rotation=45)
				ax.grid(True, alpha=0.3)
			
			# Hide unused subplots
			for i in range(n_subplots, len(axes)):
				axes[i].set_visible(False)
			
		else:
			# Single plot
			fig, ax = plt.subplots(figsize=(14, 8))
			
			# Create discrete intervals
			percent_min = cycle_df['percent_explored'].min()
			percent_max = cycle_df['percent_explored'].max()
			bins = np.linspace(percent_min, percent_max, n_intervals + 1)
			labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}%' for i in range(n_intervals)]
			
			# Add interval categories
			cycle_df_copy = cycle_df.copy()
			cycle_df_copy['interval'] = pd.cut(cycle_df_copy['percent_explored'], bins=bins, labels=labels)
			
			# Box plot with strip plot overlay
			sns.boxplot(data=cycle_df_copy, x='interval', y=metric_col, hue=group_col, ax=ax, palette='Set2')
			sns.stripplot(data=cycle_df_copy, x='interval', y=metric_col, hue=group_col, 
							ax=ax, size=3, alpha=0.4, dodge=True)
			
			ax.set_xlabel('% Explored Range')
			ax.set_ylabel(ylabel)
			ax.set_title(title)
			ax.tick_params(axis='x', rotation=45)
			ax.grid(True, alpha=0.3)
			
			# Handle duplicate legends from box + strip plots
			handles, labels_legend = ax.get_legend_handles_labels()
			n_groups = len(cycle_df[group_col].unique())
			ax.legend(handles[:n_groups], labels_legend[:n_groups])
		
		plt.tight_layout()
		plt.savefig(viz_dir / filename, dpi=300, bbox_inches='tight')
		plt.close()
		
	except Exception as e:
		self.console.print(f"[yellow]Warning: Could not create boxplot intervals plot {filename}: {e}[/yellow]")
		import traceback
		traceback.print_exc()

# Method 3: Individual Trajectories + Bold Mean Overlay
def _create_individual_trajectories_plot(self, cycle_df: pd.DataFrame, metric_col: str, group_col: str,
										title: str, ylabel: str, viz_dir: Path, filename: str,
										subplot_configs: list = None, individual_alpha: float = 0.15,
										mean_linewidth: float = 4):
	"""Plot individual experiment trajectories with bold group means overlay."""
	try:
		import numpy as np
		
		if subplot_configs:
			# Multi-subplot plot
			n_subplots = len(subplot_configs)
			cols = min(3, n_subplots)
			rows = (n_subplots + cols - 1) // cols
			
			fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
			if n_subplots == 1:
				axes = [axes]
			elif rows == 1:
				axes = axes if hasattr(axes, '__len__') else [axes]
			else:
				axes = axes.flatten()
			
			for i, config in enumerate(subplot_configs):
				if i >= len(axes):
					break
				
				ax = axes[i]
				metric = config['metric']
				
				# Plot individual experiment trajectories (faint background)
				for exp_id in cycle_df['experiment_id'].unique():
					exp_data = cycle_df[cycle_df['experiment_id'] == exp_id].sort_values('percent_explored')
					if exp_data.empty or metric not in exp_data.columns:
						continue
					
					group = exp_data[group_col].iloc[0]
					color_idx = hash(group) % 10
					ax.plot(exp_data['percent_explored'], exp_data[metric], 
							alpha=individual_alpha, linewidth=1, color=plt.cm.tab10(color_idx))
				
				# Calculate and plot bold group means
				bins = np.linspace(cycle_df['percent_explored'].min(), cycle_df['percent_explored'].max(), 15)
				bin_centers = (bins[:-1] + bins[1:]) / 2
				
				for group in sorted(cycle_df[group_col].unique()):
					group_data = cycle_df[cycle_df[group_col] == group]
					if group_data.empty or metric not in group_data.columns:
						continue
					
					# Bin and aggregate for smooth mean trajectory
					group_data_copy = group_data.copy()
					group_data_copy['bin'] = pd.cut(group_data_copy['percent_explored'], bins=bins, labels=bin_centers)
					mean_trajectory = group_data_copy.groupby('bin')[metric].mean().reset_index()
					mean_trajectory = mean_trajectory.dropna()
					
					if len(mean_trajectory) > 1:
						ax.plot(mean_trajectory['bin'], mean_trajectory[metric], 
								label=group, linewidth=mean_linewidth, marker='o', markersize=8, zorder=5)
				
				ax.set_xlabel('Cycle Number')
				ax.set_ylabel(config['ylabel'])
				ax.set_title(config['title'])
				ax.grid(True, alpha=0.3)
				
				# Compact legend configuration for EF plots
				if 'EF' in config.get('title', ''):
					ax.legend(fontsize=8, frameon=True, fancybox=False, 
								shadow=False, framealpha=0.9, edgecolor='gray',
								columnspacing=0.5, handletextpad=0.3,
								bbox_to_anchor=(1.02, 1), loc='upper left')
				else:
					ax.legend()
			
			# Hide unused subplots
			for i in range(n_subplots, len(axes)):
				axes[i].set_visible(False)
			
		else:
			# Single plot
			fig, ax = plt.subplots(figsize=(14, 10))
			
			# Plot individual experiment trajectories (faint background)
			for exp_id in cycle_df['experiment_id'].unique():
				exp_data = cycle_df[cycle_df['experiment_id'] == exp_id].sort_values('cycle')
				if exp_data.empty or metric_col not in exp_data.columns:
					continue
				
				group = exp_data[group_col].iloc[0]
				color_idx = hash(group) % 10
				ax.plot(exp_data['cycle'], exp_data[metric_col], 
						alpha=individual_alpha, linewidth=1, color=plt.cm.tab10(color_idx))
			
			# Calculate and plot bold group means by cycle
			for group in sorted(cycle_df[group_col].unique()):
				group_data = cycle_df[cycle_df[group_col] == group]
				if group_data.empty or metric_col not in group_data.columns:
					continue
				
				# Aggregate by cycle for mean trajectory
				mean_trajectory = group_data.groupby('cycle')[metric_col].mean().reset_index()
				mean_trajectory = mean_trajectory.dropna()
				
				if len(mean_trajectory) > 1:
					ax.plot(mean_trajectory['cycle'], mean_trajectory[metric_col], 
							label=group, linewidth=mean_linewidth, marker='o', markersize=8, zorder=5)
			
			ax.set_xlabel('Cycle Number')
			ax.set_ylabel(ylabel)
			ax.set_title(title)
			ax.grid(True, alpha=0.3)
			
			# Compact legend configuration for EF plots
			if 'EF' in title or 'Enrichment Factor' in title:
				ax.legend(fontsize=8, frameon=True, fancybox=False, 
							shadow=False, framealpha=0.9, edgecolor='gray',
							columnspacing=0.5, handletextpad=0.3, 
							bbox_to_anchor=(1.02, 1), loc='upper left')
			else:
				ax.legend()
		
		plt.tight_layout()
		plt.savefig(viz_dir / filename, dpi=300, bbox_inches='tight')
		plt.close()
		
	except Exception as e:
		self.console.print(f"[yellow]Warning: Could not create individual trajectories plot {filename}: {e}[/yellow]")
		import traceback
		traceback.print_exc()

# Method 4: Violin Plots for Distribution Analysis
def _create_violin_distribution_plot(self, cycle_df: pd.DataFrame, metric_col: str, group_col: str,
									title: str, ylabel: str, viz_dir: Path, filename: str,
									n_intervals: int = 4, subplot_configs: list = None):
	"""Create violin plots showing distribution shapes at different % explored levels."""
	try:
		import numpy as np
		
		if subplot_configs:
			# Multi-subplot plot
			n_subplots = len(subplot_configs)
			cols = min(3, n_subplots)
			rows = (n_subplots + cols - 1) // cols
			
			fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
			if n_subplots == 1:
				axes = [axes]
			elif rows == 1:
				axes = axes if hasattr(axes, '__len__') else [axes]
			else:
				axes = axes.flatten()
			
			for i, config in enumerate(subplot_configs):
				if i >= len(axes):
					break
				
				ax = axes[i]
				metric = config['metric']
				
				# Create discrete intervals for violin grouping
				percent_min = cycle_df['percent_explored'].min()
				percent_max = cycle_df['percent_explored'].max()
				bins = np.linspace(percent_min, percent_max, n_intervals + 1)
				labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}%' for i in range(n_intervals)]
				
				# Add interval categories
				cycle_df_copy = cycle_df.copy()
				cycle_df_copy['interval'] = pd.cut(cycle_df_copy['percent_explored'], bins=bins, labels=labels)
				
				# Violin plot
				sns.violinplot(data=cycle_df_copy, x='interval', y=metric, hue=group_col, 
								ax=ax, palette='viridis', inner='quart', linewidth=1.2)
				ax.set_xlabel('% Explored Range')
				ax.set_ylabel(config['ylabel'])
				ax.set_title(config['title'])
				ax.tick_params(axis='x', rotation=45)
				ax.grid(True, alpha=0.3)
			
			# Hide unused subplots
			for i in range(n_subplots, len(axes)):
				axes[i].set_visible(False)
			
		else:
			# Single plot
			fig, ax = plt.subplots(figsize=(14, 8))
			
			# Create discrete intervals for violin grouping
			percent_min = cycle_df['percent_explored'].min()
			percent_max = cycle_df['percent_explored'].max()
			bins = np.linspace(percent_min, percent_max, n_intervals + 1)
			labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}%' for i in range(n_intervals)]
			
			# Add interval categories
			cycle_df_copy = cycle_df.copy()
			cycle_df_copy['interval'] = pd.cut(cycle_df_copy['percent_explored'], bins=bins, labels=labels)
			
			# Violin plot
			sns.violinplot(data=cycle_df_copy, x='interval', y=metric_col, hue=group_col, 
							ax=ax, palette='viridis', inner='box', linewidth=1.2)
			
			ax.set_xlabel('% Explored Range')
			ax.set_ylabel(ylabel)
			ax.set_title(title)
			ax.tick_params(axis='x', rotation=45)
			ax.grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.savefig(viz_dir / filename, dpi=300, bbox_inches='tight')
		plt.close()
		
	except Exception as e:
		self.console.print(f"[yellow]Warning: Could not create violin distribution plot {filename}: {e}[/yellow]")
		import traceback
		traceback.print_exc()

# A) Top K recovery vs % explored plots
def _create_top_k_recovery_plots(self, cycle_df: pd.DataFrame, viz_dir: Path, plot_style: str = 'binned_statistical'):
	"""Create Top K recovery vs % explored plots using specified visualization style."""
	self.console.print(f"  • Creating Top K recovery plots ({plot_style} style)...")
	
	# Define Top K configurations
	top_k_configs = [
		{'metric': 'top_0_1_percent_overlap_mean', 'title': 'Top 0.1% Recovery', 'ylabel': 'Top 0.1% Recovery'},
		{'metric': 'top_1_percent_overlap_mean', 'title': 'Top 1% Recovery', 'ylabel': 'Top 1% Recovery'},
		{'metric': 'top_10_percent_overlap_mean', 'title': 'Top 10% Recovery', 'ylabel': 'Top 10% Recovery'},
		{'metric': 'top_100_overlap_mean', 'title': 'Top 100 Recovery', 'ylabel': 'Top 100 Recovery'},
		{'metric': 'top_1000_overlap_mean', 'title': 'Top 1000 Recovery', 'ylabel': 'Top 1000 Recovery'}
	]
	
	# Create three versions: by model type, selection strategy, initial strategy
	for group_col, group_name in [('learner_type', 'model_type'), 
									('selection_strategy', 'selection_strategy'),
									('initial_strategy', 'initial_strategy')]:
		
		filename = f'top_k_recovery_by_{group_name}_{plot_style}.png'
		title = f'Top K Recovery vs % Explored (by {group_name})'
		
		if plot_style == 'binned_statistical':
			self._create_binned_statistical_plot(
				cycle_df, None, group_col, title, 'Recovery',
				viz_dir, filename, subplot_configs=top_k_configs
			)
		elif plot_style == 'boxplot':
			self._create_boxplot_intervals_plot(
				cycle_df, None, group_col, title, 'Recovery',
				viz_dir, filename, subplot_configs=top_k_configs
			)
		elif plot_style == 'individual_trajectories':
			self._create_individual_trajectories_plot(
				cycle_df, None, group_col, title, 'Recovery',
				viz_dir, filename, subplot_configs=top_k_configs
			)
		elif plot_style == 'violin':
			self._create_violin_distribution_plot(
				cycle_df, None, group_col, title, 'Recovery',
				viz_dir, filename, subplot_configs=top_k_configs
			)
		else:
			self.console.print(f"[yellow]Warning: Unknown plot style '{plot_style}', using binned_statistical[/yellow]")
			self._create_binned_statistical_plot(
				cycle_df, None, group_col, title, 'Recovery',
				viz_dir, filename, subplot_configs=top_k_configs
			)

# B) EF vs % explored plots
def _create_ef_vs_explored_plots(self, cycle_df: pd.DataFrame, viz_dir: Path, plot_style: str = 'binned_statistical'):
	"""Create EF vs % explored plots with ground truth lines using specified visualization style."""
	self.console.print(f"  • Creating EF vs % explored plots ({plot_style} style)...")
	
	# Define EF configurations with ground truth
	ef_configs = [
		{'metric': 'ef_0_1_mean', 'title': 'EF 0.1%', 'ylabel': 'EF 0.1%', 'ground_truth_col': 'ground_truth_ef_0_1_mean'},
		{'metric': 'ef_0_5_mean', 'title': 'EF 0.5%', 'ylabel': 'EF 0.5%', 'ground_truth_col': 'ground_truth_ef_0_5_mean'},
		{'metric': 'ef_1_0_mean', 'title': 'EF 1.0%', 'ylabel': 'EF 1.0%', 'ground_truth_col': 'ground_truth_ef_1_0_mean'},
		{'metric': 'ef_5_0_mean', 'title': 'EF 5.0%', 'ylabel': 'EF 5.0%', 'ground_truth_col': 'ground_truth_ef_5_0_mean'}
	]
	
	# Create three versions: by model type, selection strategy, initial strategy
	for group_col, group_name in [('learner_type', 'model_type'), 
									('selection_strategy', 'selection_strategy'),
									('initial_strategy', 'initial_strategy')]:
		
		filename = f'ef_vs_explored_by_{group_name}_{plot_style}.png'
		title = f'Enrichment Factor vs % Explored (by {group_name})'
		
		if plot_style == 'binned_statistical':
			self._create_binned_statistical_plot(
				cycle_df, None, group_col, title, 'Enrichment Factor',
				viz_dir, filename, subplot_configs=ef_configs
			)
		elif plot_style == 'boxplot':
			self._create_boxplot_intervals_plot(
				cycle_df, None, group_col, title, 'Enrichment Factor',
				viz_dir, filename, subplot_configs=ef_configs
			)
		elif plot_style == 'individual_trajectories':
			self._create_individual_trajectories_plot(
				cycle_df, None, group_col, title, 'Enrichment Factor',
				viz_dir, filename, subplot_configs=ef_configs
			)
		elif plot_style == 'violin':
			self._create_violin_distribution_plot(
				cycle_df, None, group_col, title, 'Enrichment Factor',
				viz_dir, filename, subplot_configs=ef_configs
			)
		else:
			self.console.print(f"[yellow]Warning: Unknown plot style '{plot_style}', using binned_statistical[/yellow]")
			self._create_binned_statistical_plot(
				cycle_df, None, group_col, title, 'Enrichment Factor',
				viz_dir, filename, subplot_configs=ef_configs
			)

# C-G) Performance metrics vs % explored plots
def _create_performance_vs_explored_plots(self, cycle_df: pd.DataFrame, viz_dir: Path, plot_style: str = 'binned_statistical'):
	"""Create performance metric vs % explored plots using specified visualization style."""
	self.console.print(f"  • Creating performance vs explored plots ({plot_style} style)...")
	
	# Define performance metrics
	performance_metrics = [
		('rmse_mean', 'RMSE vs % Explored', 'RMSE', 'rmse_vs_explored'),
		('mae_mean', 'MAE vs % Explored', 'MAE', 'mae_vs_explored'),
		('mse_mean', 'MSE vs % Explored', 'MSE', 'mse_vs_explored'),
		('spearman_correlation_mean', 'Spearman vs % Explored', 'Spearman Correlation', 'spearman_vs_explored'),
		('r2_score_mean', 'R² vs % Explored', 'R² Score', 'r2_vs_explored')
	]
	
	for metric_col, title_base, ylabel, filename_base in performance_metrics:
		# Create three versions: by model type, selection strategy, initial strategy
		for group_col, group_name in [('learner_type', 'model_type'), 
										('selection_strategy', 'selection_strategy'),
										('initial_strategy', 'initial_strategy')]:
			
			filename = f'{filename_base}_by_{group_name}_{plot_style}.png'
			title = f'{title_base} (by {group_name})'
			
			if plot_style == 'binned_statistical':
				self._create_binned_statistical_plot(
					cycle_df, metric_col, group_col, title, ylabel,
					viz_dir, filename
				)
			elif plot_style == 'boxplot':
				self._create_boxplot_intervals_plot(
					cycle_df, metric_col, group_col, title, ylabel,
					viz_dir, filename
				)
			elif plot_style == 'individual_trajectories':
				self._create_individual_trajectories_plot(
					cycle_df, metric_col, group_col, title, ylabel,
					viz_dir, filename
				)
			elif plot_style == 'violin':
				self._create_violin_distribution_plot(
					cycle_df, metric_col, group_col, title, ylabel,
					viz_dir, filename
				)
			else:
				self.console.print(f"[yellow]Warning: Unknown plot style '{plot_style}', using binned_statistical[/yellow]")
				self._create_binned_statistical_plot(
					cycle_df, metric_col, group_col, title, ylabel,
					viz_dir, filename
				)

# H) Avg Score vs % explored plots
def _create_avg_score_vs_explored_plots(self, cycle_df: pd.DataFrame, viz_dir: Path, plot_style: str = 'binned_statistical'):
	"""Create average score vs % explored plots using specified visualization style."""
	self.console.print(f"  • Creating avg score vs explored plots ({plot_style} style)...")
	
	# Create three versions: by model type, selection strategy, initial strategy
	for group_col, group_name in [('learner_type', 'model_type'), 
									('selection_strategy', 'selection_strategy'),
									('initial_strategy', 'initial_strategy')]:
		
		filename = f'avg_score_vs_explored_by_{group_name}_{plot_style}.png'
		title = f'Average Score vs % Explored (by {group_name})'
		
		if plot_style == 'binned_statistical':
			self._create_binned_statistical_plot(
				cycle_df, 'avg_score_selected_mean', group_col, title, 'Average Score',
				viz_dir, filename
			)
		elif plot_style == 'boxplot':
			self._create_boxplot_intervals_plot(
				cycle_df, 'avg_score_selected_mean', group_col, title, 'Average Score',
				viz_dir, filename
			)
		elif plot_style == 'individual_trajectories':
			self._create_individual_trajectories_plot(
				cycle_df, 'avg_score_selected_mean', group_col, title, 'Average Score',
				viz_dir, filename
			)
		elif plot_style == 'violin':
			self._create_violin_distribution_plot(
				cycle_df, 'avg_score_selected_mean', group_col, title, 'Average Score',
				viz_dir, filename
			)
		else:
			self.console.print(f"[yellow]Warning: Unknown plot style '{plot_style}', using binned_statistical[/yellow]")
			self._create_binned_statistical_plot(
				cycle_df, 'avg_score_selected_mean', group_col, title, 'Average Score',
				viz_dir, filename
			)

# I) Intra-batch diversity vs % explored plots
def _create_diversity_vs_explored_plots(self, cycle_df: pd.DataFrame, viz_dir: Path, plot_style: str = 'binned_statistical'):
	"""Create intra-batch diversity vs % explored plots using specified visualization style."""
	self.console.print(f"  • Creating diversity vs explored plots ({plot_style} style)...")
	
	# Create three versions: by model type, selection strategy, initial strategy
	for group_col, group_name in [('learner_type', 'model_type'), 
									('selection_strategy', 'selection_strategy'),
									('initial_strategy', 'initial_strategy')]:
		
		filename = f'diversity_vs_explored_by_{group_name}_{plot_style}.png'
		title = f'Intra-batch Diversity vs % Explored (by {group_name})'
		
		if plot_style == 'binned_statistical':
			self._create_binned_statistical_plot(
				cycle_df, 'intra_batch_diversity_mean', group_col, title, 'Intra-batch Diversity',
				viz_dir, filename
			)
		elif plot_style == 'boxplot':
			self._create_boxplot_intervals_plot(
				cycle_df, 'intra_batch_diversity_mean', group_col, title, 'Intra-batch Diversity',
				viz_dir, filename
			)
		elif plot_style == 'individual_trajectories':
			self._create_individual_trajectories_plot(
				cycle_df, 'intra_batch_diversity_mean', group_col, title, 'Intra-batch Diversity',
				viz_dir, filename
			)
		elif plot_style == 'violin':
			self._create_violin_distribution_plot(
				cycle_df, 'intra_batch_diversity_mean', group_col, title, 'Intra-batch Diversity',
				viz_dir, filename
			)
		else:
			self.console.print(f"[yellow]Warning: Unknown plot style '{plot_style}', using binned_statistical[/yellow]")
			self._create_binned_statistical_plot(
				cycle_df, 'intra_batch_diversity_mean', group_col, title, 'Intra-batch Diversity',
				viz_dir, filename
			)

# L) Uncertainty vs % explored plots
def _create_uncertainty_vs_explored_plots(self, cycle_df: pd.DataFrame, viz_dir: Path, plot_style: str = 'binned_statistical'):
	"""Create uncertainty vs % explored plots using specified visualization style."""
	self.console.print(f"  • Creating uncertainty vs explored plots ({plot_style} style)...")
	
	# Filter for models that have uncertainty
	uncertainty_models = cycle_df[cycle_df['uncertainty_mean_mean'].notna()]
	if uncertainty_models.empty:
		self.console.print("    [yellow]No uncertainty data available, skipping uncertainty plots[/yellow]")
		return
	
	# Create three versions: by model type, selection strategy, initial strategy
	for group_col, group_name in [('learner_type', 'model_type'), 
									('selection_strategy', 'selection_strategy'),
									('initial_strategy', 'initial_strategy')]:
		
		filename = f'uncertainty_vs_explored_by_{group_name}_{plot_style}.png'
		title = f'Uncertainty vs % Explored (by {group_name})'
		
		if plot_style == 'binned_statistical':
			self._create_binned_statistical_plot(
				uncertainty_models, 'uncertainty_mean_mean', group_col, title, 'Mean Uncertainty',
				viz_dir, filename
			)
		elif plot_style == 'boxplot':
			self._create_boxplot_intervals_plot(
				uncertainty_models, 'uncertainty_mean_mean', group_col, title, 'Mean Uncertainty',
				viz_dir, filename
			)
		elif plot_style == 'individual_trajectories':
			self._create_individual_trajectories_plot(
				uncertainty_models, 'uncertainty_mean_mean', group_col, title, 'Mean Uncertainty',
				viz_dir, filename
			)
		elif plot_style == 'violin':
			self._create_violin_distribution_plot(
				uncertainty_models, 'uncertainty_mean_mean', group_col, title, 'Mean Uncertainty',
				viz_dir, filename
			)
		else:
			self.console.print(f"[yellow]Warning: Unknown plot style '{plot_style}', using binned_statistical[/yellow]")
			self._create_binned_statistical_plot(
				uncertainty_models, 'uncertainty_mean_mean', group_col, title, 'Mean Uncertainty',
				viz_dir, filename
			)

# J) Parameter Combination Performance Matrix
def _create_parameter_matrix_plot(self, cycle_df: pd.DataFrame, viz_dir: Path):
	"""Create heatmap showing final performance for all learner-strategy pairs."""
	self.console.print("  • Creating parameter combination matrix...")
	
	try:
		# Get final cycle data for each experiment
		final_cycle_data = []
		for exp_id, exp_data in cycle_df.groupby('experiment_id'):
			if not exp_data.empty:
				final_row = exp_data.iloc[-1]  # Get last cycle
				final_cycle_data.append(final_row)
		
		if not final_cycle_data:
			return
		
		final_df = pd.DataFrame(final_cycle_data)
		
		# Create pivot table for heatmap
		metrics_to_plot = ['r2_score_mean', 'rmse_mean', 'ef_1_0_mean', 'spearman_correlation_mean']
		
		for metric in metrics_to_plot:
			if metric in final_df.columns:
				# Create pivot table
				pivot_data = final_df.pivot_table(
					values=metric, 
					index='learner_type', 
					columns='selection_strategy', 
					aggfunc='mean'
				)
				
				# Create heatmap
				fig, ax = plt.subplots(figsize=(10, 8))
				sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f', 
							cbar_kws={'label': metric.replace('_mean', '').replace('_', ' ').title()})
				plt.title(f'Parameter Combination Matrix - {metric.replace("_mean", "").replace("_", " ").title()}')
				plt.xlabel('Selection Strategy')
				plt.ylabel('Learner Type')
				plt.tight_layout()
				plt.savefig(viz_dir / f'parameter_matrix_{metric.replace("_mean", "")}.png', 
							dpi=300, bbox_inches='tight')
				plt.close()
		
	except Exception as e:
		self.console.print(f"[yellow]Warning: Could not create parameter matrix: {e}[/yellow]")

# K) Uncertainty vs Performance Trajectories
def _create_uncertainty_performance_trajectories(self, cycle_df: pd.DataFrame, viz_dir: Path):
	"""Create uncertainty vs performance trajectories for uncertainty-enabled models."""
	self.console.print("  • Creating uncertainty vs performance trajectories...")
	
	try:
		# Filter for models with uncertainty
		uncertainty_models = cycle_df[cycle_df['uncertainty_mean_mean'].notna()]
		if uncertainty_models.empty:
			self.console.print("    [yellow]No uncertainty data available, skipping trajectory plots[/yellow]")
			return
		
		# Create trajectory plots for different performance metrics
		performance_metrics = ['r2_score_mean', 'ef_1_0_mean', 'spearman_correlation_mean']
		
		for metric in performance_metrics:
			if metric in uncertainty_models.columns:
				fig, ax = plt.subplots(figsize=(10, 8))
				
				# Plot trajectories for each experiment
				for exp_id, exp_data in uncertainty_models.groupby('experiment_id'):
					exp_data = exp_data.sort_values('cycle')
					if len(exp_data) > 1:
						# Color by learner type
						learner = exp_data['learner_type'].iloc[0]
						strategy = exp_data['selection_strategy'].iloc[0]
						
						ax.plot(exp_data['uncertainty_mean_mean'], exp_data[metric], 
								'o-', alpha=0.7, label=f'{learner}_{strategy}' if exp_id == uncertainty_models['experiment_id'].iloc[0] else "",
								markersize=4, linewidth=1.5)
				
				ax.set_xlabel('Mean Uncertainty')
				ax.set_ylabel(metric.replace('_mean', '').replace('_', ' ').title())
				ax.set_title(f'Uncertainty vs {metric.replace("_mean", "").replace("_", " ").title()} Trajectories')
				ax.grid(True, alpha=0.3)
				# Only show legend for first few entries to avoid clutter
				handles, labels = ax.get_legend_handles_labels()
				if len(labels) <= 10:
					ax.legend()
				
				plt.tight_layout()
				plt.savefig(viz_dir / f'uncertainty_vs_{metric.replace("_mean", "")}_trajectories.png', 
							dpi=300, bbox_inches='tight')
				plt.close()
		
	except Exception as e:
		self.console.print(f"[yellow]Warning: Could not create uncertainty trajectories: {e}[/yellow]")

def save_results_csv(self, include_cycle_by_cycle: bool = True):
	"""Save results in CSV format for further analysis.
	
	Args:
		include_cycle_by_cycle: Whether to also generate cycle-by-cycle CSV export
	"""
	if not self.results:
		return
	
	# Flatten results for CSV export with unified data model
	csv_data = []
	for result in self.results:
		if 'error' in result or result.get('skipped', False):
			continue
		
		# Base experiment information
		row = {
			'experiment_id': result['experiment_id'],
			'experiment_name': result['experiment_name'],
			'learner_type': result['parameters']['learner_type'],
			'score_direction': result['parameters'].get('score_direction', 'higher'),
			'pruning_strategy': result['parameters'].get('pruning_strategy', None),
			'learner_strategy_optimal': result.get('learner_strategy_optimal', True),
			'n_successful_repeats': result['n_successful_repeats'],
			'n_failed_repeats': result['n_failed_repeats']
		}
		
		# Unified cycle information (available for both traditional and custom experiments)
		row.update({
			'experiment_type': result.get('experiment_type', 'traditional'),
			'custom_cycle_spec': result.get('custom_cycle_spec', ''),
			'total_cycles': result.get('total_cycles', 0),
			'unique_strategies': ','.join(result.get('unique_strategies', [])),
			'primary_strategy': result.get('primary_strategy', ''),
			'strategy_summary': result.get('strategy_summary', ''),
			'batch_size_pattern': result.get('batch_size_pattern', 'unknown'),
			'n_strategy_transitions': len(result.get('strategy_transitions', []))
		})
		
		# Legacy fields (for backward compatibility)
		row.update({
			'n_cycles': result['parameters'].get('n_cycles'),
			'batch_size_fraction': result['parameters'].get('batch_size_fraction'),
			'selection_strategy': result['parameters'].get('selection_strategy'),
			'initial_strategy': result['parameters'].get('initial_strategy')
		})
		
		# Add all metrics (excluding cycle_by_cycle_metrics to keep traditional format clean)
		for key, value in result.items():
			if (key not in row and not key.startswith('parameters') and 
				key != 'errors' and key != 'cycle_by_cycle_metrics'):
				row[key] = value
		
		csv_data.append(row)
	
	if csv_data:
		csv_df = pd.DataFrame(csv_data)
		csv_file = self.output_dir / 'parameter_sweep_results.csv'
		csv_df.to_csv(csv_file, index=False)
		self.console.print(f"[green]Results saved to {csv_file}[/green]")
		
		# NEW: Export cycle-by-cycle data if requested
		if include_cycle_by_cycle:
			self.export_cycle_by_cycle_csv()

def export_cycle_by_cycle_csv(self):
	"""Export cycle-by-cycle results for learning curve analysis."""
	if not self.results:
		return
	
	cycle_data = []
	for result in self.results:
		if 'error' in result or result.get('skipped', False):
			continue
		
		# Skip if no cycle-by-cycle data
		if 'cycle_by_cycle_metrics' not in result:
			continue
		
		experiment_info = {
			'experiment_id': result['experiment_id'],
			'experiment_name': result['experiment_name'],
			'learner_type': result['parameters']['learner_type'],
			'score_direction': result['parameters'].get('score_direction', 'higher'),
			'n_successful_repeats': result['n_successful_repeats'],
			# Unified cycle information
			'experiment_type': result.get('experiment_type', 'traditional'),
			'custom_cycle_spec': result.get('custom_cycle_spec', ''),
			'primary_strategy': result.get('primary_strategy', ''),
			'strategy_summary': result.get('strategy_summary', ''),
			# Legacy fields (for backward compatibility)
			'selection_strategy': result['parameters'].get('selection_strategy'),
			'initial_strategy': result['parameters'].get('initial_strategy')
		}
		
		# Process each cycle
		for cycle_metrics in result['cycle_by_cycle_metrics']:
			row = experiment_info.copy()
			
			# Add cycle-specific data
			for key, value in cycle_metrics.items():
				row[key] = value
			
			cycle_data.append(row)
	
	if cycle_data:
		cycle_df = pd.DataFrame(cycle_data)
		cycle_csv_file = self.output_dir / 'parameter_sweep_cycle_by_cycle.csv'
		cycle_df.to_csv(cycle_csv_file, index=False)
		self.console.print(f"[green]Cycle-by-cycle results saved to {cycle_csv_file}[/green]")
		
		# Generate learning curve summary statistics
		self._generate_learning_curve_summary(cycle_df)

def _generate_learning_curve_summary(self, cycle_df: pd.DataFrame):
	"""Generate summary statistics for learning curves.
	
	Args:
		cycle_df: DataFrame with cycle-by-cycle results
	"""
	try:
		if cycle_df.empty:
			return
		
		summary_stats = []
		
		# Group by experiment for analysis
		for exp_id, exp_data in cycle_df.groupby('experiment_id'):
			if len(exp_data) < 2:  # Need at least 2 cycles for trend analysis
				continue
			
			exp_info = exp_data.iloc[0]
			
			# Calculate learning curve metrics
			stats = {
				'experiment_id': exp_id,
				'experiment_name': exp_info['experiment_name'],
				'learner_type': exp_info['learner_type'],
				'selection_strategy': exp_info['selection_strategy'],
				'n_cycles': len(exp_data),
				'n_successful_repeats': exp_info['n_successful_repeats']
			}
			
			# Analyze key performance metrics if available
			for metric in ['r2_score_mean', 'rmse_mean', 'ef_1_0_mean', 'spearman_correlation_mean']:
				if metric in exp_data.columns:
					values = exp_data[metric].dropna()
					if len(values) >= 2:
						stats[f'{metric}_initial'] = values.iloc[0]
						stats[f'{metric}_final'] = values.iloc[-1]
						stats[f'{metric}_improvement'] = values.iloc[-1] - values.iloc[0]
						stats[f'{metric}_max'] = values.max()
						stats[f'{metric}_convergence_cycle'] = values.idxmax()
			
			summary_stats.append(stats)
		
		if summary_stats:
			summary_df = pd.DataFrame(summary_stats)
			summary_file = self.output_dir / 'learning_curve_summary.csv'
			summary_df.to_csv(summary_file, index=False)
			self.console.print(f"[green]Learning curve summary saved to {summary_file}[/green]")
			
	except Exception as e:
		self.console.print(f"[yellow]Warning: Could not generate learning curve summary: {e}[/yellow]")
