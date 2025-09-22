import 'package:flutter/material.dart';
import '../models.dart';

class ReceiptCard extends StatelessWidget {
  final UploadResult res;
  final String? selectedCategory;
  final ValueChanged<String?>? onCategoryChanged;

  const ReceiptCard({
    super.key,
    required this.res,
    this.selectedCategory,
    this.onCategoryChanged,
  });

  static const cats = ['Utilities','Food','Groceries','Transportation','Health & Wellness','Others'];

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(res.store ?? 'Unknown Store', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 6),
            Text('Total: ${res.total?.toStringAsFixed(2) ?? '—'}'),
            Text('Date: ${res.date ?? '—'}'),
            Text('Predicted: ${res.category ?? 'No model'}'),
            if (res.confidence != null)
              Text('Confidence: ${(res.confidence! * 100).toStringAsFixed(1)}%'),
            const SizedBox(height: 8),
            DropdownButtonFormField<String>(
              initialValue: selectedCategory ?? res.category ?? 'Others',
              items: cats.map((c) => DropdownMenuItem(value: c, child: Text(c))).toList(),
              onChanged: onCategoryChanged,
              decoration: const InputDecoration(labelText: 'Correct / Confirm Category'),
            ),
            const SizedBox(height: 8),
            ExpansionTile(
              title: const Text('OCR Text'),
              children: [
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Text(res.text),
                ),
              ],
            ),
            const Divider(height: 20),
            Text(
              'Engines — YOLO: ${res.yoloUsed} | OCR.space: ${res.ocrSpaceUsed} (${res.ocrSource ?? 'n/a'})',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}
