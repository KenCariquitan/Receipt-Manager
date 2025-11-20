import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers.dart';
import '../models.dart';
import 'receipt_detail_page.dart';

class ReceiptsPage extends ConsumerStatefulWidget {
  const ReceiptsPage({super.key});

  @override
  ConsumerState<ReceiptsPage> createState() => _ReceiptsPageState();
}

class _ReceiptsPageState extends ConsumerState<ReceiptsPage> {
  final TextEditingController _searchController = TextEditingController();
  String _searchQuery = '';
  String? _categoryFilter;
  DateTimeRange? _dateRange;

  @override
  void initState() {
    super.initState();
    _searchController.addListener(() {
      setState(() {
        _searchQuery = _searchController.text.trim().toLowerCase();
      });
    });
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  Future<void> _pickDateRange() async {
    final now = DateTime.now();
    final initial = _dateRange ??
        DateTimeRange(
          start: DateTime(now.year, now.month, 1),
          end: now,
        );
    final picked = await showDateRangePicker(
      context: context,
      firstDate: DateTime(now.year - 5),
      lastDate: DateTime(now.year + 5),
      initialDateRange: initial,
    );
    if (picked != null) {
      setState(() => _dateRange = picked);
    }
  }

  void _clearFilters() {
    setState(() {
      _searchController.clear();
      _searchQuery = '';
      _categoryFilter = null;
      _dateRange = null;
    });
  }

  List<Receipt> _applyFilters(List<Receipt> list) {
    return list.where((r) {
      final storeText = (r.store ?? r.storeNormalized ?? '').toLowerCase();
      final matchesSearch =
          _searchQuery.isEmpty || storeText.contains(_searchQuery);

      final categoryValue = r.category?.trim();
      final matchesCategory = _categoryFilter == null ||
          _categoryFilter == 'All' ||
          (_categoryFilter != null &&
              categoryValue != null &&
              categoryValue == _categoryFilter);

      bool matchesDate = true;
      if (_dateRange != null) {
        final parsed = r.date != null ? DateTime.tryParse(r.date!) : null;
        matchesDate = parsed != null &&
            !parsed.isBefore(_dateRange!.start) &&
            !parsed.isAfter(_dateRange!.end);
      }

      return matchesSearch && matchesCategory && matchesDate;
    }).toList();
  }

  Widget _buildFilters(List<Receipt> receipts) {
    final categories = <String>{};
    for (final r in receipts) {
      final cat = r.category?.trim();
      if (cat != null && cat.isNotEmpty) {
        categories.add(cat);
      }
    }
    final categoryItems = ['All', ...categories.toList()..sort()];
    final dateLabel = _dateRange == null
        ? 'Any date'
        : '${MaterialLocalizations.of(context).formatShortDate(_dateRange!.start)} - '
            '${MaterialLocalizations.of(context).formatShortDate(_dateRange!.end)}';

    return Card(
      margin: const EdgeInsets.all(12),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            TextField(
              controller: _searchController,
              decoration: InputDecoration(
                labelText: 'Search store',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: _searchQuery.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () => _searchController.clear(),
                      )
                    : null,
              ),
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              value: _categoryFilter ?? 'All',
              items: categoryItems
                  .map(
                    (cat) => DropdownMenuItem<String>(
                      value: cat,
                      child: Text(cat),
                    ),
                  )
                  .toList(),
              decoration: const InputDecoration(
                labelText: 'Category',
                prefixIcon: Icon(Icons.category_outlined),
              ),
              onChanged: (val) {
                setState(() {
                  if (val == null || val == 'All') {
                    _categoryFilter = null;
                  } else {
                    _categoryFilter = val;
                  }
                });
              },
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _pickDateRange,
                    icon: const Icon(Icons.date_range),
                    label: Text(dateLabel),
                  ),
                ),
                if (_dateRange != null)
                  Padding(
                    padding: const EdgeInsets.only(left: 8),
                    child: IconButton(
                      tooltip: 'Clear date filter',
                      icon: const Icon(Icons.clear),
                      onPressed: () => setState(() => _dateRange = null),
                    ),
                  ),
              ],
            ),
            if (_searchQuery.isNotEmpty ||
                _categoryFilter != null ||
                _dateRange != null)
              Align(
                alignment: Alignment.centerRight,
                child: TextButton.icon(
                  onPressed: _clearFilters,
                  icon: const Icon(Icons.refresh),
                  label: const Text('Reset filters'),
                ),
              ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final recs = ref.watch(receiptsProvider);
    return Scaffold(
      body: recs.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => Center(child: Text('Error: $e')),
        data: (list) {
          if (list.isEmpty) {
            return const Center(child: Text('No receipts yet.'));
          }
          final filtered = _applyFilters(list);
          return Column(
            children: [
              _buildFilters(list),
              Expanded(
                child: filtered.isEmpty
                    ? const Center(
                        child:
                            Text('No receipts match the selected filters.'),
                      )
                    : RefreshIndicator(
                        onRefresh: () async =>
                            ref.refresh(receiptsProvider.future),
                        child: ListView.separated(
                          physics: const AlwaysScrollableScrollPhysics(),
                          itemCount: filtered.length,
                          separatorBuilder: (_, __) =>
                              const Divider(height: 1),
                          itemBuilder: (_, i) {
                            final r = filtered[i];
                            return ListTile(
                              title: Text(r.store ?? 'Unknown'),
                              subtitle: Text(
                                  '${r.date ?? '-'} • ${r.category ?? '-'}'),
                              trailing: Text(r.total != null
                                  ? '₱${r.total!.toStringAsFixed(2)}'
                                  : '-'),
                              onTap: () {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (_) =>
                                        ReceiptDetailPage(receipt: r),
                                  ),
                                ).then(
                                  (_) =>
                                      ref.refresh(receiptsProvider.future),
                                );
                              },
                            );
                          },
                        ),
                      ),
              ),
            ],
          );
        },
      ),
    );
  }
}
