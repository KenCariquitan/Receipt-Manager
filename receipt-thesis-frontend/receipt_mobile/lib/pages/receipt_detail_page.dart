import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../models.dart';
import '../api_client.dart';
final apiProvider = Provider<ApiClient>((ref) => ApiClient());

class ReceiptDetailPage extends ConsumerStatefulWidget {
  final Receipt receipt;
  const ReceiptDetailPage({super.key, required this.receipt});

  @override
  ConsumerState<ReceiptDetailPage> createState() => _ReceiptDetailPageState();
}

class _ReceiptDetailPageState extends ConsumerState<ReceiptDetailPage> {
  late final TextEditingController storeCtl;
  late final TextEditingController dateCtl;
  late final TextEditingController totalCtl;
  String? category;

  static const cats = ['Utilities','Food','Groceries','Transportation','Health & Wellness','Others'];

  @override
  void initState() {
    super.initState();
    storeCtl = TextEditingController(text: widget.receipt.store ?? '');
    dateCtl  = TextEditingController(text: widget.receipt.date ?? '');
    totalCtl = TextEditingController(text: widget.receipt.total?.toStringAsFixed(2) ?? '');
    category = widget.receipt.category ?? 'Others';
  }

  @override
  void dispose() {
    storeCtl.dispose(); dateCtl.dispose(); totalCtl.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final api = ref.read(apiProvider);
    final t = double.tryParse(totalCtl.text);
    final ok = await api.updateReceipt(
      id: widget.receipt.id,
      store: storeCtl.text.isEmpty ? null : storeCtl.text,
      date: dateCtl.text.isEmpty ? null : dateCtl.text,
      total: t,
      category: category,
    );
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(ok ? 'Saved' : 'Failed')));
      if (ok) Navigator.pop(context);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Receipt Detail')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: ListView(
          children: [
            TextField(controller: storeCtl, decoration: const InputDecoration(labelText: 'Store')),
            const SizedBox(height: 12),
            TextField(controller: dateCtl, decoration: const InputDecoration(labelText: 'Date (YYYY-MM-DD)')),
            const SizedBox(height: 12),
            TextField(controller: totalCtl, decoration: const InputDecoration(labelText: 'Total (₱)'), keyboardType: TextInputType.number),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              initialValue: category,
              items: cats.map((c) => DropdownMenuItem(value: c, child: Text(c))).toList(),
              onChanged: (v) => setState(() => category = v),
              decoration: const InputDecoration(labelText: 'Category'),
            ),
            const SizedBox(height: 20),
            FilledButton(onPressed: _save, child: const Text('Save')),
          ],
        ),
      ),
    );
  }
}
