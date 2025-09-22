import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers.dart';
import 'package:fl_chart/fl_chart.dart';

class DashboardPage extends ConsumerWidget {
  const DashboardPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final summary = ref.watch(summaryProvider);
    final byCat = ref.watch(byCategoryProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Dashboard')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: ListView(
          children: [
            summary.when(
              loading: () => const LinearProgressIndicator(),
              error: (e, _) => Text('Error: $e'),
              data: (s) => Wrap(
                spacing: 16, runSpacing: 16,
                children: [
                  _statCard('Total Spend', '₱${s.totalSpend.toStringAsFixed(2)}'),
                  _statCard('Receipts', s.totalReceipts.toString()),
                  _statCard('MTD Spend', '₱${s.monthToDateSpend.toStringAsFixed(2)}'),
                  _statCard('Top Category', s.topCategory ?? '—'),
                ],
              ),
            ),
            const SizedBox(height: 20),
            Text('Spending by Category', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            byCat.when(
              loading: () => const LinearProgressIndicator(),
              error: (e, _) => Text('Error: $e'),
              data: (rows) {
                if (rows.isEmpty) return const Text('No data yet.');
                final bars = rows.map((r) => BarChartGroupData(
                  x: rows.indexOf(r),
                  barRods: [
                    BarChartRodData(toY: (r['total'] as num).toDouble(), width: 18),
                  ],
                )).toList();
                final labels = rows.map((r) => (r['category'] ?? 'Unknown') as String).toList();
                return SizedBox(
                  height: 220,
                  child: BarChart(BarChartData(
                    borderData: FlBorderData(show: false),
                    titlesData: FlTitlesData(
                      leftTitles: const AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 38)),
                      bottomTitles: AxisTitles(sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, meta) {
                          final i = value.toInt();
                          return Padding(
                            padding: const EdgeInsets.only(top: 6),
                            child: Text(i >= 0 && i < labels.length ? labels[i] : '', style: const TextStyle(fontSize: 10)),
                          );
                        },
                      )),
                      rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                      topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    ),
                    barGroups: bars,
                  )),
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
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(title, style: const TextStyle(fontSize: 12, color: Colors.grey)),
            const SizedBox(height: 6),
            Text(value, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          ]),
        ),
      ),
    );
  }
}
