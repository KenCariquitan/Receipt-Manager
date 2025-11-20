import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models.dart';
import '../providers.dart';

class DashboardPage extends ConsumerWidget {
  const DashboardPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final summary = ref.watch(summaryProvider);
    final byCat = ref.watch(byCategoryProvider);
    final topMerchants = ref.watch(topMerchantsProvider);
    final weekday = ref.watch(weekdaySpendProvider);
    final rolling = ref.watch(rolling30Provider);
    // Removed lowConfidence section for cleaner dashboard; provider not watched here.

    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: ListView(
          children: [
            summary.when(
              loading: () => const LinearProgressIndicator(),
              error: (e, _) => Text('Error: $e'),
              data: (s) => Wrap(
                spacing: 16,
                runSpacing: 16,
                children: [
                  _statCard('Total Spend', _formatCurrency(s.totalSpend)),
                  _statCard('Receipts', s.totalReceipts.toString()),
                  _statCard('MTD Spend', _formatCurrency(s.monthToDateSpend)),
                  _statCard('Top Category', s.topCategory ?? '-'),
                ],
              ),
            ),
            const SizedBox(height: 20),
            Text('Spending by Category',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            byCat.when(
              loading: () => const LinearProgressIndicator(),
              error: (e, _) => Text('Error: $e'),
              data: (rows) {
                if (rows.isEmpty) return const Text('No data yet.');
                final bars = <BarChartGroupData>[];
                final labels = <String>[];
                for (var i = 0; i < rows.length; i++) {
                  final row = rows[i];
                  bars.add(
                    BarChartGroupData(
                      x: i,
                      barRods: [
                        BarChartRodData(
                          toY: (row['total'] as num).toDouble(),
                          width: 18,
                        )
                      ],
                    ),
                  );
                  labels.add((row['category'] ?? 'Unknown') as String);
                }
                return SizedBox(
                  height: 220,
                  child: BarChart(
                    BarChartData(
                      borderData: FlBorderData(show: false),
                      gridData: const FlGridData(drawHorizontalLine: true),
                      titlesData: FlTitlesData(
                        leftTitles: const AxisTitles(
                          sideTitles: SideTitles(
                            showTitles: true,
                            reservedSize: 38,
                          ),
                        ),
                        bottomTitles: AxisTitles(
                          sideTitles: SideTitles(
                            showTitles: true,
                            getTitlesWidget: (value, meta) {
                              final i = value.toInt();
                              return Padding(
                                padding: const EdgeInsets.only(top: 6),
                                child: Text(
                                  i >= 0 && i < labels.length ? labels[i] : '',
                                  style: const TextStyle(fontSize: 10),
                                ),
                              );
                            },
                          ),
                        ),
                        rightTitles: const AxisTitles(
                          sideTitles: SideTitles(showTitles: false),
                        ),
                        topTitles: const AxisTitles(
                          sideTitles: SideTitles(showTitles: false),
                        ),
                      ),
                      barGroups: bars,
                    ),
                  ),
                );
              },
            ),
            const SizedBox(height: 24),
            Text('Top Merchants (This Month)',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            topMerchants.when(
              loading: () => const LinearProgressIndicator(),
              error: (e, _) => Text('Error: $e'),
              data: (rows) {
                if (rows.isEmpty) {
                  return const Text('No purchases recorded this month.');
                }
                return Card(
                  child: Column(
                    children: rows
                        .map(
                          (m) => ListTile(
                            leading: CircleAvatar(
                              child: Text(
                                _initialForStore(m.store),
                                style: const TextStyle(
                                    fontWeight: FontWeight.bold),
                              ),
                            ),
                            title: Text(m.store),
                            subtitle: Text(
                              '${m.receiptCount} receipt${m.receiptCount == 1 ? '' : 's'}',
                            ),
                            trailing: Text(_formatCurrency(m.totalSpend)),
                          ),
                        )
                        .toList(),
                  ),
                );
              },
            ),
            const SizedBox(height: 24),
            Text('Weekday Spend',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            weekday.when(
              loading: () => const LinearProgressIndicator(),
              error: (e, _) => Text('Error: $e'),
              data: (rows) {
                if (rows.isEmpty) return const Text('No data yet.');
                final bars = rows
                    .map(
                      (r) => BarChartGroupData(
                        x: r.weekday,
                        barRods: [
                          BarChartRodData(toY: r.totalSpend, width: 18),
                        ],
                      ),
                    )
                    .toList();
                return SizedBox(
                  height: 220,
                  child: BarChart(
                    BarChartData(
                      borderData: FlBorderData(show: false),
                      gridData: const FlGridData(drawHorizontalLine: true),
                      titlesData: FlTitlesData(
                        leftTitles: const AxisTitles(
                          sideTitles: SideTitles(
                            showTitles: true,
                            reservedSize: 42,
                          ),
                        ),
                        bottomTitles: AxisTitles(
                          sideTitles: SideTitles(
                            showTitles: true,
                            getTitlesWidget: (value, meta) => Padding(
                              padding: const EdgeInsets.only(top: 6),
                              child: Text(
                                _weekdayLabel(value.toInt()),
                                style: const TextStyle(fontSize: 11),
                              ),
                            ),
                          ),
                        ),
                        rightTitles: const AxisTitles(
                          sideTitles: SideTitles(showTitles: false),
                        ),
                        topTitles: const AxisTitles(
                          sideTitles: SideTitles(showTitles: false),
                        ),
                      ),
                      barGroups: bars,
                    ),
                  ),
                );
              },
            ),
            const SizedBox(height: 24),
            Text('Rolling 30-Day Spend',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            rolling.when(
              loading: () => const LinearProgressIndicator(),
              error: (e, _) => Text('Error: $e'),
              data: (rows) {
                if (rows.isEmpty) return const Text('No data yet.');
                final sorted = [...rows]
                  ..sort((a, b) => a.date.compareTo(b.date));
                final spots = <FlSpot>[];
                for (var i = 0; i < sorted.length; i++) {
                  spots.add(FlSpot(i.toDouble(), sorted[i].totalSpend));
                }
                final maxY = spots.fold<double>(
                    0.0, (prev, e) => e.y > prev ? e.y : prev);
                return SizedBox(
                  height: 220,
                  child: LineChart(
                    LineChartData(
                      minX: 0,
                      maxX: spots.isNotEmpty ? spots.last.x : 0,
                      minY: 0,
                      maxY: maxY > 0 ? maxY * 1.1 : 1,
                      gridData: const FlGridData(drawHorizontalLine: true),
                      borderData: FlBorderData(show: false),
                      titlesData: FlTitlesData(
                        leftTitles: const AxisTitles(
                          sideTitles: SideTitles(
                            showTitles: true,
                            reservedSize: 42,
                          ),
                        ),
                        bottomTitles: AxisTitles(
                          sideTitles: SideTitles(
                            showTitles: true,
                            reservedSize: 36,
                            getTitlesWidget: (value, meta) {
                              final idx = value.toInt();
                              if (idx < 0 || idx >= sorted.length) {
                                return const SizedBox.shrink();
                              }
                              final d = sorted[idx].date;
                              return Padding(
                                padding: const EdgeInsets.only(top: 6),
                                child: Text(
                                  '${d.month}/${d.day}',
                                  style: const TextStyle(fontSize: 10),
                                ),
                              );
                            },
                          ),
                        ),
                        rightTitles: const AxisTitles(
                          sideTitles: SideTitles(showTitles: false),
                        ),
                        topTitles: const AxisTitles(
                          sideTitles: SideTitles(showTitles: false),
                        ),
                      ),
                      lineBarsData: [
                        LineChartBarData(
                          spots: spots,
                          isCurved: true,
                          color: Theme.of(context).colorScheme.primary,
                          barWidth: 3,
                          dotData: const FlDotData(show: false),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _statCard(String title, String value) {
    return SizedBox(
      width: 160,
      child: Card(
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(fontSize: 12, color: Colors.grey),
              ),
              const SizedBox(height: 6),
              Text(
                value,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }


  static String _formatCurrency(double value) => '₱${value.toStringAsFixed(2)}';

  static String _weekdayLabel(int dow) {
    const names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    if (dow >= 0 && dow < names.length) return names[dow];
    return dow == 7 ? 'Sun' : '?';
  }

  static String _initialForStore(String store) {
    final trimmed = store.trim();
    if (trimmed.isEmpty) return '?';
    final chars = trimmed.characters;
    if (chars.isEmpty) {
      return trimmed[0].toUpperCase();
    }
    return chars.first.toUpperCase();
  }
}
