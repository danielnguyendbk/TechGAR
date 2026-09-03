'use client';

import { useState } from 'react';
import { Check, MapPinned, Trash2, X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { runtimeClient } from '@/lib/api/runtime-client';
import type { WorldPoint } from '@/lib/domain/types';

interface GateEditorProps {
  readonly editing: boolean;
  readonly points: readonly WorldPoint[];
  readonly onEditingChange: (editing: boolean) => void;
  readonly onPointsChange: (points: readonly WorldPoint[]) => void;
}

export function GateEditor({ editing, points, onEditingChange, onPointsChange }: GateEditorProps) {
  const [message, setMessage] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function save(): Promise<void> {
    setPending(true);
    setMessage(null);
    try {
      await runtimeClient.saveGates(points);
      setMessage('Đã lưu cấu hình cổng 6 điểm.');
      onEditingChange(false);
    } catch (caught) {
      setMessage(caught instanceof Error ? caught.message : 'Không thể lưu cấu hình cổng');
    } finally {
      setPending(false);
    }
  }

  return (
    <section className="space-y-3" aria-labelledby="gate-editor-title">
      <div>
        <h3 id="gate-editor-title" className="text-sm font-semibold">Cấu hình cổng</h3>
        <p className="mt-1 text-xs text-muted-foreground">Chọn đúng 6 điểm trên bản đồ: 3 điểm vào, 3 điểm ra.</p>
      </div>
      {!editing ? (
        <Button variant="outline" className="w-full justify-start" onClick={() => { onPointsChange([]); onEditingChange(true); setMessage(null); }}>
          <MapPinned className="size-4" /> Bắt đầu chọn điểm
        </Button>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          <Button variant="outline" onClick={() => onPointsChange([])}><Trash2 /> Xóa điểm</Button>
          <Button variant="outline" onClick={() => onEditingChange(false)}><X /> Hủy</Button>
          <Button className="col-span-2" disabled={points.length !== 6 || pending} onClick={() => void save()}>
            <Check /> {pending ? 'Đang lưu…' : `Lưu ${points.length}/6 điểm`}
          </Button>
        </div>
      )}
      {message && <output className="block rounded-lg bg-muted px-3 py-2 text-xs">{message}</output>}
    </section>
  );
}
