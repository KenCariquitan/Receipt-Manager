import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers.dart';
// import '../models.dart';
import 'receipt_detail_page.dart';

class ReceiptsPage extends ConsumerWidget {
  const ReceiptsPage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final recs = ref.watch(receiptsProvider);
    return Scaffold(
      appBar: AppBar(title: const Text('Receipts')),
      body: recs.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => Center(child: Text('Error: $e')),
        data: (list) {
          if (list.isEmpty) return const Center(child: Text('No receipts yet.'));
          return RefreshIndicator(
            onRefresh: () async => ref.refresh(receiptsProvider),
            child: ListView.separated(
              itemCount: list.length,
              separatorBuilder: (_, __) => const Divider(height: 1),
              itemBuilder: (_, i) {
                final r = list[i];
                return ListTile(
                  title: Text(r.store ?? 'Unknown'),
                  subtitle: Text('${r.date ?? '—'} • ${r.category ?? '—'}'),
                  trailing: Text(r.total != null ? '₱${r.total!.toStringAsFixed(2)}' : '—'),
                  onTap: () {
                    Navigator.push(context, MaterialPageRoute(
                      builder: (_) => ReceiptDetailPage(receipt: r),
                    )).then((_) => ref.refresh(receiptsProvider));
                  },
                );
              },
            ),
          );
        },
      ),
    );
  }
}
